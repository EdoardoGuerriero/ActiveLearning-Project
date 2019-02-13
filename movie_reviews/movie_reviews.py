#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:51:00 2019

@author: edoardoguerriero
"""
from os import listdir
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd 

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
        
# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load all docs in a directory
def process_docs(directory, data):
	# walk through all files in the folder
    n_doc = 0
    
    if 'neg' in directory:
        stance = 'NEG'
    else:
        stance = 'POS'
    
    for filename in listdir(directory):
		# skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
		# create the full path of the file to open
        path = directory + '/' + filename
		# load document
        doc = load_doc(path)
        cleaned_doc = ' '.join(clean_doc(doc))
         
        data.loc[n_doc,'first_cleaning_text'] = cleaned_doc
        data.loc[n_doc,'Stance'] = stance
         
        n_doc+=1
    
    return data
        
#%% load documents
    
# specify directory to load
directory_pos = 'txt_sentoken/pos'
directory_neg = 'txt_sentoken/neg'

data_pos = pd.DataFrame(columns=['first_cleaning_text', 'Stance'])
data_neg = pd.DataFrame(columns=['first_cleaning_text', 'Stance'])

#%% generate dataset

data_pos = process_docs(directory_pos, data_pos)
data_neg = process_docs(directory_neg, data_neg)

#%% split data into train test and unlabeled
 
complete_dataset = pd.concat([data_pos, data_neg])
complete_dataset = complete_dataset.sample(frac=1)

test_train_data = complete_dataset.head(int(complete_dataset.shape[0]/2))
unlabeled_data = complete_dataset.drop(test_train_data.index)

test_train_data.reset_index(drop=True, inplace=True)
unlabeled_data.reset_index(drop=True, inplace=True) 

#%% SPLIT TRAINING-TEST FOR FIRST TRAINING

from sklearn.model_selection import train_test_split as tts

X = test_train_data.drop(columns=['Stance']).astype('object')
y = test_train_data['Stance'].astype('object')

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.5)

#%% TRANING-TEST MODELS

from preprocessing_functions4 import train_model
from classifiers_collection import naive_bayes
from classifiers_collection import decision_tree
from classifiers_collection import logistic_regression
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(input='content', ngram_range=(1,2),lowercase=True, \
                       stop_words='english', analyzer='word', max_features=200)
X_train_vect = vect.fit_transform(X_train.values.ravel())
X_test_vect = vect.transform(X_test.values.ravel())

# multinomial naive bayes
accuracy_nb, frame_prob_y_train_nb, frame_prob_y_test_nb, nb = \
                train_model(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test, naive_bayes)
                
# decision tree
accuracy_dt, frame_prob_y_train_dt, frame_prob_y_test_dt, dt = \
                train_model(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test, decision_tree)
                
# logistic regression
accuracy_lr, frame_prob_y_train_lr, frame_prob_y_test_lr, lr = \
                train_model(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test, logistic_regression)

#%% APPLY MODELS TO UNLABELLED DATA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from preprocessing_functions4 import active_learning_test2
import re 

models = [nb]#, lr, dt] 

results_dict = {
    'least_confident': [],
    'max_margin': [],
    'entropy': [],
    'random': []
    }


sample_size = 100

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
                y_test, unlabeled_data, model, model_base, results_dict, sample_size)


