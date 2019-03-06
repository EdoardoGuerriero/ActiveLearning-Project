#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""

import os
import plac
import random
import pathlib
import cytoolz
import numpy
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import thinc.extra.datasets
from spacy.compat import pickle
import spacy
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from preprocessing_functions3 import rename_labels
from preprocessing_functions3 import cleaning_text
from preprocessing_functions3 import split_save
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping

# to remove deprecation warning
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

class SentimentAnalyser(object):
    @classmethod
    def load(cls, step, nlp, path_step, active, max_length=100):

        with open(path_step+'/model_config.json') as file_:
            model = model_from_json(file_.read())
        with open(path_step+'/model_weights.pkl', 'rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, path_step, active, max_length)

    def __init__(self, model, path_step, active='entropy', max_length=100):
        
        self._model = model
        self.path_step = path_step   
        self.active = active
        self.max_length = max_length


    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=10, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
#                if numpy.argmax(label) > 3:
#                    print(numpy.argmax(label))
                if self.active == 'entropy':
                    entr = entropy(label)
                    sent.doc.sentiment += entr  
                    
                elif self.active == 'margin':
                    margin = numpy.partition(-label, 1)
                    margin_scores = -numpy.abs(margin[:,0] - margin[:, 1])
                    sent.doc.sentiment += margin_scores
                     
                elif self.active == 'least_confident':    
                    sent.doc.sentiment += 1-numpy.max(label)
                    
                elif self.active == 'random':
                    sent.doc.sentiment += random.choice(label)
                    
                else:
                    sent.doc.sentiment += numpy.argmax(label)
                    
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs

# precision metric
def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# recall metric
def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# f1 metric
def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=10,
          nb_epoch=5, by_sentence=False):
    
    print("\nLoading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    
    print("\nParsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0,\
                               patience=2, verbose=0, mode='auto')
    
    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    history = model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              epochs=nb_epoch, batch_size=batch_size, callbacks=[early_stop])
    
    return model, history


def compile_lstm(embeddings, shape, settings):
    
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'],
                                 recurrent_dropout=settings['dropout'],
                                 dropout=settings['dropout'])))
    model.add(Dense(shape['nr_class'], activation='softmax'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='categorical_crossentropy',
		  metrics=['accuracy', precision, recall, fbeta_score])
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(texts, labels, step, max_length=100):
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(step, nlp, max_length=max_length, active=False))

    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):

        correct += bool(doc.sentiment >= 0.5) == bool(numpy.argmax(labels[i]))
        i += 1
    return float(correct) / i

def test_report(texts, labels, step, max_length=100):
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(step, nlp, max_length=max_length, active=False))
    
    
    return

def active_step(texts, step, path_step, strategy='entropy', max_length=100):

    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(step, nlp, path_step, max_length=max_length, active=strategy))
    
    data_ordered = pd.DataFrame(index=numpy.arange(len(texts)),columns=['Tweet', 'scores'])
    
    i = 0
    for doc in nlp.pipe(texts, batch_size=10, n_threads=4):
        
        data_ordered.loc[i, 'Tweet'] = doc.text
        data_ordered.loc[i, 'scores'] = doc.sentiment
        
        i += 1
        
    data_ordered.sort_values(by=['scores'], ascending=True, inplace=True)
    
    return data_ordered


def read_data(data_dir, limit=0):
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples) # Unzips into two lists

def plot_history(history, path_step):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    recall = history.history['recall']
    val_recall = history.history['val_recall']
    precision = history.history['precision']
    val_precision = history.history['val_precision']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.subplot(2, 2, 3)
    plt.plot(x, precision, 'b', label='Training precision')
    plt.plot(x, val_precision, 'r', label='Validation precision')
    plt.title('Training and validation precision')
    plt.subplot(2, 2, 4)
    plt.plot(x, recall, 'b', label='Training recall')
    plt.plot(x, val_recall, 'r', label='Validation recall')
    plt.title('Training and validation recall')
    plt.savefig(path_step+'/Training_plots.png')

@plac.annotations(
    train_dir=("Location of training file or directory"),
    dev_dir=("Location of development file or directory"),
    model_dir=("Location of output model directory",),
    is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
    nr_hidden=("Number of hidden units", "option", "H", int),
    max_length=("Maximum sentence length", "option", "L", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "i", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int)
    )


def train_with_active_learning(folder_name, target_class, target_col, \
                               text_col, classes_col,\
                               step=1, tweets_per_step=70, \
                               nr_hidden=64, max_length=100, \
                               dropout=0.5, learn_rate=0.01, \
                               nb_epoch=10, batch_size=20, nr_examples=-1):  
    
    # check if target folder exist
    path_target = './{}'.format(folder_name)
    if not os.path.exists(path_target):
        os.makedirs(path_target)
    
    path_step = path_target + '/Step_{}'.format(step)
    path_prev_step = path_target + '/Step_{}'.format(step-1)
    if not os.path.exists(path_step):
        os.makedirs(path_step)
    
    if step == 1:
        
        # read and clean tweets before to split into training and test
        test_train_data_all = pd.read_csv('train.csv', engine='python')
        test_train_data = test_train_data_all.loc[test_train_data_all[classes_col]==target_class]
        test_train_data.reset_index(drop=True, inplace=True)
        
        unlabeled_data_all = pd.read_csv('test.csv', engine='python')
        unlabeled_data = unlabeled_data_all.loc[unlabeled_data_all[classes_col]==target_class]
        unlabeled_data.reset_index(drop=True, inplace=True)
    
        #test_train_data = rename_labels(test_train_data, target_col)
        #unlabeled_data = rename_labels(unlabeled_data, target_col)
        
        test_train_data = cleaning_text(test_train_data, text_col)
        unlabeled_data = cleaning_text(unlabeled_data, text_col)
        
        X_train, y_train, X_test, y_test, unlabeled_data_texts = \
            split_save(test_train_data, unlabeled_data, test_size=0.4, \
                       step=step, text_col=text_col, target_col=target_col, path_step=path_step)

        # train lstm model
        lstm, history = train(X_train, y_train, X_test, y_test,
             {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 3},
             {'dropout': dropout, 'lr': learn_rate},
             {},
             nb_epoch=nb_epoch, batch_size=batch_size)
        weights = lstm.get_weights()
        
        # save model 
        with open(path_step+'/model_weights.pkl'.format(step), 'wb') as file_:
            pickle.dump(weights[1:], file_)
            
        with open(path_step+'/model_config.json'.format(step), 'w') as file_:
            file_.write(lstm.to_json())
            
        with open(path_step+'/history.pkl'.format(step), 'wb') as file_:
            pickle.dump(history.history, file_)         
        print('\nDone saving model')
        
        # active learning step
        data_ordered = active_step(unlabeled_data_texts, step, path_step,\
                                   strategy='entropy', max_length=100)     
        print('\nDone active learning selection')
        
        # save intermediate datasets
        indices_to_label = data_ordered.index[:tweets_per_step]
        data_to_label = unlabeled_data.loc[indices_to_label].reset_index(drop=True)
        data_to_label.to_csv(path_step+'/tweet_to_label.csv'.format(step), index=False)
        indices_unlabeled = data_ordered.index[tweets_per_step:]
        data_unlabeled_next_step = unlabeled_data.loc[indices_unlabeled].reset_index(drop=True)
        data_unlabeled_next_step.to_csv(path_step+'/unlabeled_next_step.csv'.format(step), index=False)
        print('\nDone saving datasets')
        
        # plot training history
        plot_history(history, path_step)
        
    else:
        
        # load datasets from previous step
        try:
            train_data = pd.read_csv(path_prev_step+'/train_data.csv')
            test_data = pd.read_csv(path_target+'/Step_1/test_data.csv')
            unlabeled_data = pd.read_csv(path_prev_step+'/unlabeled_next_step.csv')
            annotated_data = pd.read_csv(path_prev_step+'/tweet_to_label.csv')
        
        except:
            print("\nCan't load training files from previous step folder.")
        
        train_data_all = pd.concat([train_data.reset_index(drop=True), \
                                    annotated_data.reset_index(drop=True)],sort=True, axis=0)
        train_data_all.to_csv(path_step+'/train_data.csv'.format(step), index=False)
        # shuffle training dataset 
        train_data_all = train_data_all.sample(frac=1)
        
        X_train = train_data_all[text_col]
        y_train = train_data_all[target_col]
        X_test = test_data[text_col]
        y_test = test_data[target_col]
        
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)

        unlabeled_data_texts = unlabeled_data[text_col]
        
        # train lstm model
        lstm, history = train(X_train, y_train, X_test, y_test,
             {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 3},
             {'dropout': dropout, 'lr': learn_rate},
             {},
             nb_epoch=nb_epoch, batch_size=batch_size)
        weights = lstm.get_weights()
        
        # save model 
        with open(path_step+'/model_weights.pkl'.format(step), 'wb') as file_:
            pickle.dump(weights[1:], file_)
            
        with open(path_step+'/model_config.json'.format(step), 'w') as file_:
            file_.write(lstm.to_json())
            
        with open(path_step+'/history.pkl'.format(step), 'wb') as file_:
            pickle.dump(history.history, file_)
        print('\nDone saving model')
        
        # active learning step
        data_ordered = active_step(unlabeled_data_texts, step, path_step,\
                                   strategy='entropy', max_length=100)
        print('\nDone active learning selection')
        
        # save intermediate datasets
        indices_to_label = data_ordered.index[:tweets_per_step]
        data_to_label = unlabeled_data.loc[indices_to_label].reset_index(drop=True)
        data_to_label.to_csv(path_step+'/tweet_to_label.csv'.format(step), index=False)
        indices_unlabeled = data_ordered.index[tweets_per_step:]
        data_unlabeled_next_step = unlabeled_data.loc[indices_unlabeled].reset_index(drop=True)
        data_unlabeled_next_step.to_csv(path_step+'/unlabeled_next_step.csv'.format(step), index=False)
        print('\nDone saving datasets')
        
        # plot training history
        plot_history(history, path_step)
        
    return

if __name__ == '__main__':
    
    '''
    Main function to train an lstm model one step at time.
    
    
    INPUTS:
    
    - In order to run, the script require both files from the 2016 SemEval task 
      in its same directory (file names are test.csv and train.csv if you change 
      them then replace the names with the new ones, the test file is used here 
      as a source for unlabeled data)
    
    
    PARAMETERS:
        
    - Step: the number of the training step to perform, starting from 1
    
    - Target: the label of the subset of tweets from the SemEval task to analyze
    
              'Hillary Clinton' | 'Legalization of abortion' | 'Atheism' 
              'Climate Change is a Real Concern' | 'Feminist Movement'
    
    - Tweet_per_step: number of tweets to select with the active learning strategy
    
    - Strategy: the active learning strategy. If none, return just the predicted label
    
               'entropy' | 'max_margin' | 'least_confident' | 'random'
    
    
    OUTPUTS:
    
    - Model_config.json --> contains the parameter of the trained lstm 
    
    - Model_weights.pkl --> contains the weights of the trained lstm
    
    - History --> contains all results scores: accuracy / precision / recall / f_score
    
    - Train_data --> contains the specific dataset used to train the lstm
    
    - Test_data --> contains the dataset used to test the lstm
                    N.B: the test dataset do not refet to the test.csv file. 
                    Moreover, it is generated only once, during the first step, 
                    cause the same dataset is used to test also the next models
                    without any changing.
    
    - Tweets_to_label --> csv containing tweet and target stance of the tweets 
                          selected with the specified active learning strategy
    
    - Unlabeled_next_step --> Unlabeled data which were not selected during 
                              the current step
        
    '''
    
    steps = [2,3,4]   
    text_col = 'Tweet'
    target_col = 'Stance'
    classes_col = 'Target'
    target_class = 'Feminist Movement'
    folder_name = 'Feminism tweets'
    tweets_per_step = 70
    
    for step in steps:
        train_with_active_learning(step=step, target_class=target_class, \
                                   tweets_per_step=tweets_per_step, \
                                   target_col=target_col, text_col=text_col, \
                                   classes_col=classes_col, \
                                   folder_name=folder_name)

