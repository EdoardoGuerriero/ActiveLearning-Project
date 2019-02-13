#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 01:29:27 2019

@author: edoardoguerriero
"""

import pandas as pd

# dataset for models training
test_train_data_all = pd.read_csv('train.csv', engine='python')
#test_train_data = test_train_data.head(int(len(test_train_data)/2))
test_train_data = test_train_data_all.loc[test_train_data_all.Target=='Legalization of Abortion']
test_train_data.reset_index(drop=True, inplace=True)

# dataset for active learning
unlabelled_data_all = pd.read_csv('test.csv', engine='python')
unlabelled_data = unlabelled_data_all.loc[unlabelled_data_all.Target=='Legalization of Abortion']
unlabelled_data.reset_index(drop=True, inplace=True)

#%% CONVERT ANNOTATION LABELS

from preprocessing_functions3 import rename_labels

test_train_data = rename_labels(test_train_data)
unlabelled_data = rename_labels(unlabelled_data)

#%% REMOVE HASHTAGS URLS AND TAGS FROM TEXT

from preprocessing_functions3 import cleaning_text

test_train_data = cleaning_text(test_train_data)
unlabelled_data = cleaning_text(unlabelled_data)

#%% ADD DENSE VECTORS

# load embedding vectors 
from preprocessing_functions3 import open_glove_dictionary
from preprocessing_functions3 import add_dense_vectors

# length of the embedding vectors we want to use 
length = 25
glove_dict = open_glove_dictionary(length)

test_train_data_dense = add_dense_vectors(length, glove_dict, test_train_data)
unlabelled_data_dense = add_dense_vectors(length, glove_dict, unlabelled_data)

#%% SPLIT TRAINING-TEST FOR FIRST TRAINING

from sklearn.model_selection import train_test_split as tts

X = test_train_data_dense.drop(columns=['Stance']).astype('object')
y = test_train_data_dense['Stance'].astype('object')

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.6)

#%% TRANING-TEST MODELS

from preprocessing_functions3 import train_model
from classifiers_collection import naive_bayes
from classifiers_collection import decision_tree
from classifiers_collection import logistic_regression

#select dense columns
dense_cols = [col for col in X_train.columns if 'emb_' in col]

# multinomial naive bayes
accuracy_nb, frame_prob_y_train_nb, frame_prob_y_test_nb, nb_0 = \
                train_model(X_train[dense_cols], y_train, X_test[dense_cols], y_test, naive_bayes)
# decision tree
accuracy_dt, frame_prob_y_train_dt, frame_prob_y_test_dt, dt_0 = \
                train_model(X_train[dense_cols], y_train, X_test[dense_cols], y_test, decision_tree)
# logistic regression
accuracy_lr, frame_prob_y_train_dt, frame_prob_y_test_lr, lr_0 = \
                train_model(X_train[dense_cols], y_train, X_test[dense_cols], y_test, logistic_regression)


#%% APPLY MODELS TO UNLABELLED DATA

from preprocessing_functions3 import active_learning_test2
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import re 

models = [lr_0]#, lr, dt] 

results_dict = {
    'least_confident': [],
    'max_margin': [],
    'entropy': [],
    'random': []
}

sample_size = 70

for model in models:
    reg = re.compile("^[^\(]*") 
    name = reg.match(str(model)).group()
    
    if name == 'GaussianNB':
        model_base = GaussianNB()
    elif name == 'LogisticRegression':
        model_base = LogisticRegression()
    elif name == 'DecisionTreeClassifier':
        model_base = DecisionTreeClassifier()
    else:
        print('no alowed model')
        break  
    
    ## num_samples, indices_dict
    uncertainty_sampling_results = active_learning_test2(X_train, y_train, X_test, \
                y_test, unlabelled_data_dense, model, model_base, results_dict, sample_size)






