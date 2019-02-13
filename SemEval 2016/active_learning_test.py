#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

# dataset for models training
test_train_data_all = pd.read_csv('train.csv', engine='python')
#test_train_data = test_train_data.head(int(len(test_train_data)/2))
test_train_data = test_train_data_all.loc[test_train_data_all.Target=='Legalization of Abortion']

# dataset for active learning
unlabelled_data_all = pd.read_csv('test.csv', engine='python')
unlabelled_data = unlabelled_data_all.loc[unlabelled_data_all.Target=='Legalization of Abortion']

# change labels of opinion annotations with abbreviations
from preprocessing_functions import rename_labels

test_train_data = rename_labels(test_train_data)
unlabelled_data = rename_labels(unlabelled_data)


#%% REMOVE HASHTAGS URLS AND TAGS FROM TEXT

# remove tags, hashtags simbols, urls
from preprocessing_functions import cleaning_text

test_train_data = cleaning_text(test_train_data)
unlabelled_data = cleaning_text(unlabelled_data)


#%% PERFORM STEMMING, REMOVE STOP WORDS AND CAPITAL LETTERS

# perform stemming, remove stop words
from preprocessing_functions import stem_stop

test_train_data = stem_stop(test_train_data)
unlabelled_data = stem_stop(unlabelled_data)

test_train_data.reset_index(drop=True, inplace=True)
unlabelled_data.reset_index(drop=True, inplace=True) 

#%% SPLIT TRAINING-TEST FOR FIRST TRAINING

from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer

# divide features and target
X = test_train_data['first_cleaning_text'].astype('object')
y = test_train_data['Stance'].astype('object')

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.5)

#X_train = X_train.head(int(len(X_train)/2))
#y_train = y_train.head(int(len(X_train)/2))

# define vectorizer and apply it to training and test set
vect = TfidfVectorizer(input='content', ngram_range=(1,2),lowercase=True, \
                       analyzer='word', max_features=1000, stop_words='english') 

X_train_vect = vect.fit_transform(X_train.values.ravel())
X_test_vect = vect.transform(X_test.values.ravel())

#%% TRANING-TEST MODELS

from preprocessing_functions import train_model
from classifiers_collection import naive_bayes
from classifiers_collection import decision_tree
from classifiers_collection import logistic_regression

# multinomial naive bayes
accuracy_nb, frame_prob_y_train_nb, frame_prob_y_test_nb, nb = \
                train_model(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test, naive_bayes)
# decision tree
accuracy_dt, frame_prob_y_train_dt, frame_prob_y_test_dt, dt = \
                train_model(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test, decision_tree)
# logistic regression
accuracy_lr, frame_prob_y_train_dt, frame_prob_y_test_lr, lr = \
                train_model(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test, logistic_regression)


#%% APPLY MODELS TO UNLABELLED DATA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import re
from preprocessing_functions import active_learning_test2

'''
PARAMETERS THAT CAN BE MODIFIED

plot: if True plot the final accuracies for all the strategies

save: if True the selected tweets will be saved at each step

vect_features: number of features used by the vectorizer

vect_ngrams: number of n_grams used by the vectorizer

models: list of models that can be tested (also all at once)

    - nb = naive_bayes
    - lr = logistic regression
    - dt = decision tree
    
strategies: the strategies inside results_dict can be removed, 
            but if so set plot=False otherwise you will get an error
            
'''

save = False

plot = True

vect_features = 650

vect_ngrams = (1,3)

models = [lr]

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
    uncertainty_sampling_results = active_learning_test2(save, plot, vect_features, vect_ngrams, X_train, y_train, X_test, \
                y_test, unlabelled_data, model, model_base, results_dict, sample_size)







