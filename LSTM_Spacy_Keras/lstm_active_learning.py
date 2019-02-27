#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""

import plac
import random
import cytoolz
import numpy
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from spacy.compat import pickle
import spacy
from scipy.stats import entropy
from preprocessing_functions import rename_labels
from preprocessing_functions import cleaning_text
from preprocessing_functions import split_save

# just to remove a deprecation warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

class SentimentAnalyser(object):
    @classmethod
    def load(cls, step, nlp, active, max_length=100):

        with open('Step_{}_model_config.json'.format(step)) as file_:
            model = model_from_json(file_.read())
        with open('Step_{}_model_weights.pkl'.format(step), 'rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length, active=active)

    def __init__(self, model, max_length=100, active='entropy'):
        self._model = model
        self.max_length = max_length
        self.active = active

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_length)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):

                if self.active == 'entropy':
                    entr = entropy(label)
                    sent.doc.sentiment += entr  
                    
                elif self.active == 'max_margin':
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


def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100,
          nb_epoch=5, by_sentence=False):
    
    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    
    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              epochs=nb_epoch, batch_size=batch_size)
    return model


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
		  metrics=['accuracy'])
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


def active_step(texts, step, strategy='entropy', max_length=100):
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(step, nlp, max_length=max_length, active=strategy))
    
    data_ordered = pd.DataFrame(index=numpy.arange(len(texts)),columns=['tweet', 'scores'])
    
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):

        data_ordered.loc[i, 'tweet'] = doc.text
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


@plac.annotations(
    nr_hidden=("Number of hidden units", "option", "H", int),
    max_length=("Maximum sentence length", "option", "L", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "i", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int)
)



def main(step=1, strategy='entropy', target='Feminist Movement', tweet_per_step=70,
        nr_hidden=32, max_length=100, # Input Shape
         dropout=0.5, learn_rate=0.01, # General NN config
         nb_epoch=5, batch_size=50, nr_examples=-1):  
    
    if step == 1:
        
        # read and clean tweets before to split into training and test
        test_train_data_all = pd.read_csv('train.csv', engine='python')
        test_train_data = test_train_data_all.loc[test_train_data_all.Target==target]
        test_train_data.reset_index(drop=True, inplace=True)
        
        unlabeled_data_all = pd.read_csv('test.csv', engine='python')
        unlabeled_data = unlabeled_data_all.loc[unlabeled_data_all.Target==target]
        unlabeled_data.reset_index(drop=True, inplace=True)
    
        test_train_data = rename_labels(test_train_data)
        unlabeled_data = rename_labels(unlabeled_data)
        
        test_train_data = cleaning_text(test_train_data)
        unlabeled_data = cleaning_text(unlabeled_data)
        
        X_train, y_train, X_test, y_test, unlabeled_data_texts = \
            split_save(test_train_data, unlabeled_data, test_size=0.3, step=step)

        # train lstm model
        lstm = train(X_train, y_train, X_test, y_test,
             {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 3},
             {'dropout': dropout, 'lr': learn_rate},
             {},
             nb_epoch=nb_epoch, batch_size=batch_size)
        weights = lstm.get_weights()
        
        # save model 
        with open('Step_{}_model_weights.pkl'.format(step), 'wb') as file_:
            pickle.dump(weights[1:], file_)
        with open('Step_{}_model_config.json'.format(step), 'w') as file_:
            file_.write(lstm.to_json())
        print('Done saving')
        # active learning step
        data_ordered = active_step(unlabeled_data_texts, step, \
                                   strategy='entropy', max_length=100)
        
        print('Done active step')
        # save intermediate datasets
        indices_to_label = data_ordered.index[:tweet_per_step]
        data_to_label = unlabeled_data.loc[indices_to_label].reset_index(drop=True)
        data_to_label.to_csv('Step_{}_tweet_to_label.csv'.format(step), index=False)
        indices_unlabeled = data_ordered.index[tweet_per_step:]
        data_unlabeled_next_step = unlabeled_data.loc[indices_unlabeled].reset_index(drop=True)
        data_unlabeled_next_step.to_csv('Step_{}_unlabeled_next_step.csv'.format(step), index=False)
        print('Done saving')
        
    else:
        
        # load datasets from previous step
        train_data = pd.read_csv('Step_{}_train_data.csv'.format(step-1))
        test_data = pd.read_csv('Step_1_test_data.csv')
        unlabeled_data = pd.read_csv('Step_{}_unlabeled_next_step.csv'.format(step-1))
        annotated_data = pd.read_csv('Step_{}_tweet_to_label.csv'.format(step-1))
        
        cols = ['tweet', 'Stance']
        
        train_data_all = pd.concat([train_data.reset_index(drop=True), \
                                    annotated_data.reset_index(drop=True)],sort=True, axis=0)
        train_data_all[cols].to_csv('Step_{}_train_data.csv'.format(step), index=False)
        # shuffle training dataset 
        train_data_all = train_data_all.sample(frac=1)
        
        X_train = train_data_all.tweet
        y_train = train_data_all.Stance
        X_test = test_data.tweet
        y_test = test_data.Stance
        
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)

        unlabeled_data_texts = unlabeled_data.tweet
        
        # train lstm model
        lstm = train(X_train, y_train, X_test, y_test,
             {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 3},
             {'dropout': dropout, 'lr': learn_rate},
             {},
             nb_epoch=nb_epoch, batch_size=batch_size)
        weights = lstm.get_weights()
        
        # save model 
        with open('Step_{}_model_weights.pkl'.format(step), 'wb') as file_:
            pickle.dump(weights[1:], file_)
        with open('Step_{}_model_config.json'.format(step), 'w') as file_:
            file_.write(lstm.to_json())
        
        # active learning step
        data_ordered = active_step(unlabeled_data_texts, step, \
                                   strategy='entropy', max_length=100)
        
        # save intermediate datasets
        indices_to_label = data_ordered.index[:tweet_per_step]
        data_to_label = unlabeled_data.loc[indices_to_label].reset_index(drop=True)
        data_to_label[cols].to_csv('Step_{}_tweet_to_label.csv'.format(step), index=False)
        indices_unlabeled = data_ordered.index[tweet_per_step:]
        data_unlabeled_next_step = unlabeled_data.loc[indices_unlabeled].reset_index(drop=True)
        data_unlabeled_next_step.to_csv('Step_{}_unlabeled_next_step.csv'.format(step), index=False) 
        
    return

if __name__ == '__main__':
    
    '''
    Main function to train an lstm model one step at time.
    
    
    INPUTS:
    
    - In order to run, the scriptrequire both file from the 2016 SemEval task 
      in the same directory (called test.csv and train.csv, the test file is 
      used here as a source for unlabeled data)
    
    
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
    
    step = 4
    target = 'Feminist Movement'
    tweet_per_step = 70
    strategy = 'entropy'
    
    plac.call(main(step=step, target=target,\
                   strategy=strategy, tweet_per_step=tweet_per_step))
