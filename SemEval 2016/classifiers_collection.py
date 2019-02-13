#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:57:08 2018

@author: edoardoguerriero
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle 
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# RANDOM FOREST

def random_forest(X_train, y_train, X_test, n_estimators=10, min_samples_leaf=5, filename =None, criterion='gini', print_model_details=False, gridsearch=True, save=False):

    if gridsearch:

        tuned_parameters = [{'min_samples_leaf': [2, 4],
                     'min_samples_split' : [2, 4],
                     'max_features' : ['log2'],
                     'n_estimators':[10, 25, 50],
                     'criterion':['entropy'],
                     'max_depth': [10, 15, 20]}]
        
        rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10, scoring='roc_auc') #accuracy
        
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

    # Fit the model

    rf.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(rf.best_params_)

    if gridsearch:
        rf = rf.best_estimator_

    pred_prob_training_y = rf.predict_proba(X_train)
    pred_prob_test_y = rf.predict_proba(X_test)
    pred_training_y = rf.predict(X_train)
    pred_test_y = rf.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

    if print_model_details:
        ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]
        print('Feature importance random forest:')
        for i in range(0, len(rf.feature_importances_)):
            print(X_train.columns[ordered_indices[i]])
            print(' & ')
            print(rf.feature_importances_[ordered_indices[i]])
    
    if save:
        filename = filename
        pickle.dump(rf, open(filename, 'wb'))

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y, rf

# DEEP NEURAL NETWORK

def feedforward_neural_network(X_train, y_train, X_test, filename, hidden_layer_sizes=(100,), max_iter=500, activation='logistic', \
                               alpha=0.0001, learning_rate='costant', gridsearch=True, print_model_details=False, save=False):

    if gridsearch:
        tuned_parameters = [{'hidden_layer_sizes': [(200,100),(100,50,),(50,10),], 'activation': ['tanh'],
                                 'learning_rate': ['adaptive'], 'max_iter': [1500], 'alpha': [alpha]}]
        nn = GridSearchCV(MLPClassifier(), tuned_parameters, cv=10, scoring='roc_auc')
    else:
            # Create the model
        nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate=learning_rate, alpha=alpha)

        # Fit the model
    nn.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(nn.best_params_)

    if gridsearch:
        nn = nn.best_estimator_
    
    if save:
        filename = filename
        pickle.dump(nn, open(filename, 'wb'))

        # Apply the model
    pred_prob_training_y = nn.predict_proba(X_train)
    pred_prob_test_y = nn.predict_proba(X_test)
    pred_training_y = nn.predict(X_train)
    pred_test_y = nn.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)
    
    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

# LINEAR SUPPORT VECTOR MACHINE 

def support_vector_machine_with_kernel(X_train, y_train, X_test, filename=None, kernel='linear', C=1, max_iter=1000,\
                                       gamma=1e-3, gridsearch=True, print_model_details=False, save = False):
    
    # Create the model
    if gridsearch:
        tuned_parameters = [{'kernel': ['linear'], 'gamma': [1e-3], #'rbf', 'poly', 
                     'C': [50, 100]}]
        svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='roc_auc')
    else:
        svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000)

    # Fit the model
    svm.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(svm.best_params_)

    if gridsearch:
        svm = svm.best_estimator_
    
    if save:
        filename = filename
        pickle.dump(svm, open(filename, 'wb'))
    # Apply the model
    pred_prob_training_y = svm.predict_proba(X_train)
    pred_prob_test_y = svm.predict_proba(X_test)
    pred_training_y = svm.predict(X_train)
    pred_test_y = svm.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

# KERNEL SUPPORT VECTOR MACHINE

def support_vector_machine_without_kernel(X_train, y_train, X_test, filename=None, C=1, tol=1e-3, \
                                          max_iter=1000, gridsearch=True, print_model_details=True, save=False):
    
    # Create the model
    if gridsearch:
        tuned_parameters = [{'max_iter': [1000], 'tol': [1e-3, 1e-4],
                     'C': [0.1, 1, 10]}]
        svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=10, scoring='accuracy')
    else:
        svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

    # Fit the model
    svm.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(svm.best_params_)

    if gridsearch:
        svm = svm.best_estimator_
        
    if save:
        filename = filename
        pickle.dump(svm, open(filename, 'wb'))
    # Apply the model
    
    try:
        distance_training_platt = 1/(1+np.exp(svm.decision_function(X_train)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:,None]
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
    except Exception as e:
        pred_prob_training_y = []
        frame_prob_training_y =[]
    try:
        distance_test_platt = 1/(1+np.exp(svm.decision_function(X_test)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:,None]
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)
    except Exception as e: 
        pred_prob_test_y = []
        frame_prob_test_y = []
        
    pred_training_y = svm.predict(X_train)
    pred_test_y = svm.predict(X_test)
    
    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y, svm

# K NEAREST NEIGHBOOR 

def k_nearest_neighbor(X_train, y_train, X_test, filename, n_neighbors=5, gridsearch=True, \
                       print_model_details=False, save = False, n_jobs=-1):
    
    # Create the model
    if gridsearch:
        numbers = np.arange(90,100,1)
        tuned_parameters = [{'n_neighbors': [3,92],#numbers,
                             'algorithm': ['auto'],# 'brute', 'kd_tree', 'ball_tree'],
                             'metric': ['chebyshev', 'manhattan','minkowski'],
                             'p': [2]}]
        knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10, scoring='roc_auc')
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model
    knn.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(knn.best_params_)

    if gridsearch:
        knn = knn.best_estimator_

    if save:
        filename = filename
        pickle.dump(knn, open(filename, 'wb'))
    
    # Apply the model
    pred_prob_training_y = knn.predict_proba(X_train)
    pred_prob_test_y = knn.predict_proba(X_test)
    pred_training_y = knn.predict(X_train)
    pred_test_y = knn.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

# DECISION TREE

def decision_tree(X_train, y_train, X_test, filename=None, min_samples_leaf=50, criterion='gini', print_model_details=False,\
                  export_tree_path='./', export_tree_name='tree.dot', gridsearch=True, save=False):
    # Create the model
    if gridsearch:
        tuned_parameters = [{'min_samples_leaf': [50, 100, 200, 500],
                             'criterion':['gini', 'entropy']}]
        dtree = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
    else:
        dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

    # Fit the model

    dtree.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(dtree.best_params_)

    if gridsearch:
        dtree = dtree.best_estimator_
        
    if save:
        filename = filename
        pickle.dump(dtree, open(filename, 'wb'))
    # Apply the model
    pred_prob_training_y = dtree.predict_proba(X_train)
    pred_prob_test_y = dtree.predict_proba(X_test)
    pred_training_y = dtree.predict(X_train)
    pred_test_y = dtree.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=dtree.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

    if print_model_details:
        ordered_indices = [i[0] for i in sorted(enumerate(dtree.feature_importances_), key=lambda x:x[1], reverse=True)]
        print('Feature importance decision tree:')
        for i in range(0, len(dtree.feature_importances_)):
            print(X_train.columns[ordered_indices[i]])
            print(' & ')
            print(dtree.feature_importances_[ordered_indices[i]])
        tree.export_graphviz(dtree, out_file=export_tree_path + export_tree_name, feature_names=X_train.columns, class_names=dtree.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y, dtree

# NAIVE BAYES

def naive_bayes(X_train, y_train, X_test, filename=None, gridsearch=True, print_model_details=True, save = False):
    
    # Create the model
    if gridsearch:
        tuned_parameters = []
    
        nb = GridSearchCV(GaussianNB(), tuned_parameters, cv=10, scoring='accuracy')
    else:
        nb = GaussianNB()

    # Fit the model
    nb.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(nb.best_params_)

    if gridsearch:
        nb = nb.best_estimator_
    
    if save:
        filename = filename
        pickle.dump(nb, open(filename, 'wb'))
    # Apply the model
    pred_prob_training_y = nb.predict_proba(X_train)
    pred_prob_test_y = nb.predict_proba(X_test)
    pred_training_y = nb.predict(X_train)
    pred_test_y = nb.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y, nb

# LOGISTIC REGRESSION

def logistic_regression(X_train, y_train, X_test, filename=None, C=1, penalty='l1',\
                                       max_iter=100, gridsearch=True, print_model_details=True, save = False):
    
    # Create the model
    if gridsearch:
        tuned_parameters = [{ 'C': [0.1, 1, 1.5],
                              'penalty': ['l1', 'l2'],
                              'max_iter': [100]}]
  
        lg = GridSearchCV(LogisticRegression(), tuned_parameters, cv=10, scoring='accuracy')
    else:
        lg = LogisticRegression(C=1, penalty='l1')

    # Fit the model
    lg.fit(X_train, y_train.values.ravel())

    if gridsearch and print_model_details:
        print(lg.best_params_)

    if gridsearch:
        lg = lg.best_estimator_
    
    if save:
        filename = filename
        pickle.dump(lg, open(filename, 'wb'))
    # Apply the model
    pred_prob_training_y = lg.predict_proba(X_train)
    pred_prob_test_y = lg.predict_proba(X_test)
    pred_training_y = lg.predict(X_train)
    pred_test_y = lg.predict(X_test)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=lg.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=lg.classes_)

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y, lg












