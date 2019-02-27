#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:05:43 2019

@author: edoardoguerriero
"""

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


from sklearn.model_selection import train_test_split


# convert into tuples for lstm model training
def split_save(test_train_data, unlabeled_data, test_size, step):
    
    text = test_train_data['first_cleaning_text']
    stance = test_train_data['Stance']
    
    X_train, X_test, y_train, y_test = train_test_split(text, stance, test_size=test_size)
    
    unlabeled_data = unlabeled_data['first_cleaning_text'].values
    
    train_dataset = pd.concat([X_train.to_frame(name='tweet').reset_index(drop=True)\
                               , y_train.to_frame(name='Stance').reset_index(drop=True)],axis=1)
    test_dataset = pd.concat([X_test.to_frame(name='tweet').reset_index(drop=True)\
                              , y_test.to_frame(name='Stance').reset_index(drop=True)],axis=1)
    # save for next step
    train_dataset.to_csv('Step_{}_train_data.csv'.format(step),index=False)
    test_dataset.to_csv('Step_{}_test_data.csv'.format(step),index=False)
    
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    return X_train, y_train, X_test, y_test, unlabeled_data


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


# perform active learning selection, return the data to label to retrain the model
def active_learning_sampling(unlabelled_data, probabilities, kind='naive'):
    
    if kind == 'naive':
        
        # find max probability value for each row 
        probabilities['max_proba'] = probabilities[['EO','WO','NO']].max(axis=1)
        
        # sort probabilities
        probabilities = probabilities.sort_values(by='max_proba', axis=0)
        
        # select unlabelled instances with low probability score
        train_data_new = unlabelled_data.loc[probabilities.head(1300).index]
        
    if kind == 'random':
        
        # select unlabelled instances by sampling randomly from unlabelled data
        train_data_new = unlabelled_data.sample(n=1300)
    
    return train_data_new

from copy import deepcopy
from active_learning import ActiveLearner
import seaborn as sns
import matplotlib.pyplot as plt


def active_learning_test(X_labeled, y_labeled, X_test, y_test, X_unlabeled, clf, results_dict, range_queries):
    
    uncertainty_sampling_results = deepcopy(results_dict)
    dense_cols = [col for col in X_labeled.columns if 'emb_' in col]
    
    # number of raws we're adding at each step, initialize with 0 because of the baseline accuracy
    # with no raws added
    num_samples = [0]
    
    # train model to get the baseline accuracy
    clf.fit(X_labeled[dense_cols], y_labeled.values.ravel())    
    accuracy_0 = np.sum(clf.predict(X_test[dense_cols]) == y_test) / np.shape(X_test)[0]
    print('Baseline', accuracy_0)
    # dictionary to store the index of the selected tweets
    indices_dict = {}
    
    for strategy in uncertainty_sampling_results:
        print(strategy)
        
        oracle = ActiveLearner(strategy=strategy)
        
        # create sub-dictionary for the strategy 
        indices_dict[strategy] = {}
        # add baseline accuracy to the results dictionary
        uncertainty_sampling_results[strategy].append(accuracy_0)
        
        # copy initial datasets and model
        clf_strategy = clf
        X_augmented = X_labeled
        y_augmented = y_labeled
        unlabeled_reduced = X_unlabeled[dense_cols]
        
        # create sub-dictionary for each training step (for the indices)
        for i in range(len(range_queries)):
            indices_dict[strategy]['Step_{}'.format(i+1)] = {}
        
        # counter used to keep trace of the steps
        counter = 1
        for num_queries in range_queries:
            print('Step ', counter)
            # add number of rows selected (for the final plot)
            num_samples.append(num_queries*counter)

            # in the first step there is nothing to remove from the unlabeled dataset
            # in the next steps we need to remove the tweets we chose in the previous step
            if counter != 1:
                unlabeled_reduced.drop(unlabeled_reduced.index\
                                       [[indices_dict[strategy]['Step_{}'.format(counter-1)]]], inplace=True)
            if strategy == 'random':
                print(unlabeled_reduced.shape)
            
            # perform active learning, store the indices ofthe tweets in the dictionary
            step_indices = oracle.rank(clf_strategy, unlabeled_reduced, num_queries)
            indices_dict[strategy]['Step_{}'.format(counter)]= step_indices
            
            #print(set(unlabeled_reduced.index).intersection(set(step_indices)))        
            #print(set(indices_dict[strategy]['Step_{}'.format(counter)]) <= set(list(unlabeled_reduced.index)))
            
            temp_X = pd.DataFrame(columns = X_labeled.columns)
            temp_y = pd.Series()
            temp_X2 = X_unlabeled.Tweet.loc[indices_dict[strategy]['Step_{}'.format(counter)]]
            temp_X2.to_csv('Selected_tweets_{}_Step_{}.csv'.format(strategy, counter))
            
            i = 0
            for index in indices_dict[strategy]['Step_{}'.format(counter)]:
                temp_y.loc[i] = X_unlabeled['Stance'].loc[index]
                temp_X.loc[i] = X_unlabeled.loc[index]
                i+=1
            
            # concatenate the initial dataset and the new tweets
            X_augmented = pd.concat([X_augmented[dense_cols], temp_X[dense_cols]], sort=False)
            y_augmented = pd.concat([y_augmented, temp_y], sort=False)
            
            clf_strategy = clf_strategy.fit(X_augmented.sample(frac=1).values, y_augmented.values.ravel())
            accuracy = np.sum(clf_strategy.predict(X_test[dense_cols]) == y_test) / np.shape(X_test)[0]
            uncertainty_sampling_results[strategy].append(accuracy)
            
            print(np.sum(X_augmented.duplicated()))
            print('Accuracy: ',accuracy)
                
            counter+=1
    #print('data ', len(uncertainty_sampling_results['random']))   
    print('x ', len(num_samples))
    fig = plt.figure()
    sns.set_style("darkgrid")
    plt.plot(\
             #np.arange(11), uncertainty_sampling_results['random'], 'red', 
             np.arange(11), uncertainty_sampling_results['least_confident'], 'blue',
             np.arange(11), uncertainty_sampling_results['max_margin'], 'green',
             np.arange(11), uncertainty_sampling_results['entropy'], 'orange',)
    plt.legend(['Least Confident', 'Max Margin', 'Entropy'], loc=4)
    plt.ylabel('Accuracy') 
    plt.xlabel('Number of training steps ({} tweets more each step)'.format(range_queries[0])) 
    plt.title('Tweets opinion labelling')
    #plt.ylim([0.5,0.7])
    fig.savefig('Feminism2_tweets_dataset.png')
    plt.show()
    plt.close(fig)

    
    return uncertainty_sampling_results, num_samples, indices_dict


from Active_strategies import active_selection
import copy

def active_learning_test2(X_labeled, y_labeled, X_test, y_test, X_unlabeled, clf, clf_base, results_dict, sample_size):
    
    uncertainty_sampling_results = deepcopy(results_dict)
    dense_cols = [col for col in X_labeled.columns if 'emb_' in col]
    target_col = 'Stance'
    
    # train model to get the baseline accuracy
    clf.fit(X_labeled[dense_cols], y_labeled.values.ravel())    
    accuracy_0 = np.sum(clf.predict(X_test[dense_cols]) == y_test) / np.shape(X_test)[0]
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
#            print('Step ', counter)
            
            X_augmented, y_augmented, unlabeled_reduced = active_selection\
                (strategy, sample_size, dense_cols,\
                 target_col, X_augmented, y_augmented, clf_strategy, unlabeled_reduced)
            
            clf_to_train = copy.copy(clf_base)
            clf_strategy = clf_to_train.fit(X_augmented[dense_cols].sample(frac=1).values, y_augmented.values.ravel())
            accuracy = np.sum(clf_strategy.predict(X_test[dense_cols]) == y_test) / X_test.shape[0]
            uncertainty_sampling_results[strategy].append(accuracy)

#            print(np.sum(X_augmented[dense_cols].duplicated()))
#            print('Accuracy: ',accuracy)
                
            counter+=1
    
    x_axis_values = np.zeros(len(uncertainty_sampling_results[strategy]))
    for i in np.arange(len(uncertainty_sampling_results[strategy])):
        x_axis_values[i] = sample_size*i
    
    #x_axis_values = np.asarray(x_axis_values)
    #x_axis_values = np.arange(0,(X_unlabeled.shape[0]), sample_size)
    
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
    #plt.ylim([0.5,0.7])
    fig.savefig('Movies_dataset.png')
    plt.show()
    plt.close(fig)

    return uncertainty_sampling_results

