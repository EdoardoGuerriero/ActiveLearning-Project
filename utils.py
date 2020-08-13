#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:53:09 2019

@author: edoardoguerriero
"""
import pickle
import dill
import numpy as np 

# functions to save and load histories of models trainings
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# functions to save and load torchtext Field objects TEXT and LABELS
def save_field(field, name):
    with open(name+'.Field', 'wb')as f:
     dill.dump(field,f)
    
def load_field(name):
    with open(name + '.Field', 'rb')as f:
     return dill.load(f)

# functions to save and load torchtext Field objects TEXT and LABELS
def save_iterator(field, name):
    with open(name+'.data', 'wb')as f:
     dill.dump(field,f)
    
def load_iterator(name):
    with open(name + '.data', 'rb')as f:
     return dill.load(f)


def count_parameters(model):
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters:,} trainable parameters')
    return parameters


def get_top_n_features(vectorizer, clf, class_labels, num_top):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    features_dict = {}
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-num_top:]
        features_dict[class_label] = [feature_names[j] for j in top_n]
        
    return features_dict

# function to open GloVe embedding vectors as dictionary
def load_Glove(filename):
    f = open(filename,'r')
    dict_ = {}
    
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        dict_[word] = embedding
        
    return dict_