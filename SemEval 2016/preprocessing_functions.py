#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re 
from urllib.parse import urlparse


# remove tags, hashtags simbol, url
def cleaning_text(data):
    first_cleaned_text = []
    number_hashtags = []
    list_hashtags = []
    
    for t, tweet in data.iterrows():
        text = tweet.Tweet
        hashtags = re.findall(r"#(\w+)", str(text))
        number_hashtags.append(len(hashtags))
        list_hashtags.append(hashtags)
        
        for w, word in enumerate(hashtags):
            text = text.replace(('#'+word), word)
        new_string = ''
        for i in text.split():
            s, n, p, pa, q, f = urlparse(i)
            if s and n:
                pass
            elif i[:1] == '@':
                pass
            else:
                new_string = new_string.strip() + ' ' + i
        first_cleaned_text.append(new_string)
    
    data['first_cleaning_text'] = pd.Series(data=first_cleaned_text, index=data.index)
    data['n_hashtags'] = pd.Series(data=number_hashtags, index=data.index)
    data['hashtags'] = pd.Series(data=list_hashtags, index=data.index)
    
    return data


# change long sentence used to annotate the opinions with an abbreviation
# EO = explicit opinion -- WO = weak opinion -- NO = not an opinion
def rename_labels(data):
    
    labels = data['Opinion Towards'].unique()
    
    for label in labels:
        
        if label[0] == '1':
            data = data.replace(label, 'EO')
        elif label[0] == '2':
            data = data.replace(label, 'WO')
        else:
            data = data.replace(label, 'NO')
            
    return data

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# perforn stemmin and remove stop words
def stem_stop(data):
    stemmer = SnowballStemmer("english")
    stopwords_list = stopwords.words("english")
    stopwords_list.remove('not')
    
    data['cleaned_text'] = data['first_cleaning_text'].apply(lambda x: " ".join\
        ([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stopwords_list]).lower())
    
    data = data.dropna()
    
    return data


from classifiers_collection import naive_bayes
from sklearn.metrics import classification_report as clfr
from sklearn.metrics import accuracy_score


# read embedding vectors and store them in a dictionary from glove file
# length refers to the desired length of the embedding vectors to use
def open_glove_dictionary(length):

    embeddings_index = {}
    f = open('glove.twitter.27B.'+str(length)+'d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    return embeddings_index


# add embedding vectors to the data by creating n columns with n equal to 
# the length of the embedded vectors 
def add_dense_vectors(length, embedding_dictionary, data_init):
    
    data = data_init
    
    # initialize columns to save the values of embedding vectors 
    for i in range(0, length):
        
        data['emb_{}'.format(i)] = pd.Series(index=data.index)
        
    for r, row in data.iterrows():
        
        text = row.first_cleaning_text        
        embedding_vector = np.zeros(length)
        
        n_words = 0
        for w, word in enumerate(text.split()):
            
            # retrive embedding vector for each word, sum them
            try:
                embedding_vector = embedding_vector + embedding_dictionary.get(word)
                n_words+=1 
            except:
                n_words+=1
                 
        # devide to get the average 
        #embedding_vector = np.divide(embedding_vector, n_words)
        
        for i in range(0, length):
            
            # all zeros for words not included in the embedded vectors' file.
            try:  
                data.at[r, 'emb_{}'.format(i)] = embedding_vector[i]
            except:
                data.at[r, 'emb_{}'.format(i)] = 0
    
    return data


# first training of the model using the labelled data avaiable
def train_model(X_train, y_train, X_test, y_test, classifier):
    
    # train model, return predictions, probabilities and model
    pred_y_train, pred_y_test, frame_prob_y_train, frame_prob_y_test, model = \
    classifier(X_train, y_train, X_test, gridsearch=False)
    
    accuracy = accuracy_score(y_test, pred_y_test)
    
    # print report (accuracy, precision, recall, f1 score)
    reg = re.compile("^[^\(]*")  # regular expression to print the name of the classifier
    print('\n', reg.match(str(model)).group())
    print('Test accuracy :', accuracy)
    print('Scores after first training: \n', clfr(y_test, pred_y_test, target_names=model.classes_))
    
    return accuracy, frame_prob_y_train, frame_prob_y_test, model


from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt


from Active_strategies import active_selection
import copy
from sklearn.feature_extraction.text import TfidfVectorizer

def active_learning_test2(save, plot, n_features, n_grams, X_labeled, y_labeled, X_test, y_test, X_unlabeled, clf, clf_base, results_dict, sample_size):
    
    uncertainty_sampling_results = deepcopy(results_dict)
    train_col = 'first_cleaning_text'
    target_col = 'Stance'
    
    # train model to get the baseline accuracy
    vect = TfidfVectorizer(input='content', ngram_range=n_grams, lowercase=True, \
                           analyzer='word', max_features=n_features, stop_words='english') 
    X_train_vect = vect.fit_transform(X_labeled.values.ravel())
    X_test_vect = vect.transform(X_test.values.ravel())
    
    clf.fit(X_train_vect.toarray(), y_labeled.values.ravel())    
    accuracy_0 = np.sum(clf.predict(X_test_vect.toarray()) == y_test) / np.shape(y_test)[0]
    print('Baseline', accuracy_0)
    
    for strategy in uncertainty_sampling_results:
        print(strategy)
        
        # add baseline accuracy to the results dictionary
        uncertainty_sampling_results[strategy].append(accuracy_0)
        
        # copy initial datasets and model
        clf_strategy = copy.copy(clf)
        X_augmented = X_labeled.copy()
        y_augmented = y_labeled.copy()
        unlabeled_reduced = X_unlabeled.copy()
        
        # counter used to keep trace of the steps
        counter = 1
        while sample_size <= unlabeled_reduced.shape[0]:
            
            # perform active learning selection
            X_augmented, y_augmented, unlabeled_reduced = active_selection\
                (save, counter, vect, strategy, sample_size, train_col,\
                 target_col, X_augmented, y_augmented, clf_strategy, unlabeled_reduced)
            
            # reset vectorizer and retrain it on the new training dataset
            vect = TfidfVectorizer(input='content', ngram_range=n_grams, lowercase=True,\
                                   analyzer='word', max_features=n_features, stop_words='english') 
            
            X_train_vect = vect.fit_transform(X_augmented.values.ravel())
            X_test_vect = vect.transform(X_test.values.ravel())
            
            # retrain the model and store the test accuracy
            clf_to_train = copy.copy(clf_base)
            clf_strategy = clf_to_train.fit(X_train_vect.toarray(), y_augmented.values.ravel())
            accuracy = np.sum(clf_strategy.predict(X_test_vect.toarray()) == y_test) / y_test.shape[0]
            uncertainty_sampling_results[strategy].append(accuracy)

            counter+=1
    
    x_axis_values = np.zeros(len(uncertainty_sampling_results[strategy]))
    for i in np.arange(len(uncertainty_sampling_results[strategy])):
        x_axis_values[i] = sample_size*i

    if plot:
        fig = plt.figure()
        sns.set_style("darkgrid")
        plt.plot(\
                 x_axis_values, uncertainty_sampling_results['random'], 'red', 
                 x_axis_values, uncertainty_sampling_results['least_confident'], 'blue',
                 x_axis_values, uncertainty_sampling_results['max_margin'], 'green',
                 x_axis_values, uncertainty_sampling_results['entropy'], 'orange',)
        plt.legend(['Random','Least Confident', 'Max Margin', 'Entropy'], loc=4)
        plt.ylabel('Accuracy') 
        plt.xlabel('Number of training steps ({} tweets more each step)'.format(sample_size)) 
        plt.title('Feminism tweets stance detection')
        fig.savefig('Feminism_{}.png'.format(sample_size))
        plt.show()
        plt.close(fig)

    return uncertainty_sampling_results

