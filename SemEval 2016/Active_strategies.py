#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.stats import entropy
import numpy as np
import pandas as pd
import re

def active_selection(save, step, vect, strategy, sample_size, train_col, target_col, X_train, y_train, model, unlabeled_data):
    
    train_col = 'first_cleaning_text'
    
    reg = re.compile("^[^\(]*") 
    name = reg.match(str(model)).group()
    
    filename = '{}_{}_Step_{}.csv'.format(name, strategy, step)
     
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    unlabeled_data.reset_index(inplace=True, drop=True)
    
    unlabeled_vect = vect.transform(unlabeled_data[train_col].values.ravel())
    probs = pd.DataFrame(data=(model.predict_proba(unlabeled_vect.toarray())))
    
    if strategy == 'random':
        
        probs = probs.sample(frac=1)
        unlabeled_data = unlabeled_data.reindex(probs.index)
        df_temp = unlabeled_data.head(sample_size)
        
        X_expanded = pd.DataFrame(data=np.concatenate([X_train.values.reshape((X_train.shape[0],1)), df_temp[train_col].values.reshape((sample_size, 1))], axis=0), columns = [train_col]) 
        y_expanded = pd.concat([y_train, df_temp[target_col]], sort=True)
        unlabeled_reduced = unlabeled_data.drop(df_temp.head(sample_size).index)
        
    elif strategy == 'max_margin':
        
        margin = np.partition(-probs, 1, axis=1)
        margin_scores = pd.DataFrame(data=-np.abs(margin[:,0] - margin[:, 1]))
        margin_scores.sort_values(by=margin_scores.columns[0], ascending=True, inplace=True)
        unlabeled_data = unlabeled_data.reindex(margin_scores.index)
        df_temp = unlabeled_data.head(sample_size)
        
        X_expanded = pd.DataFrame(data=np.concatenate([X_train.values.reshape((X_train.shape[0],1)), df_temp[train_col].values.reshape((sample_size, 1))], axis=0), columns = [train_col]) 
        y_expanded = pd.concat([y_train, df_temp[target_col]], sort=True)
        unlabeled_reduced = unlabeled_data.drop(df_temp.head(sample_size).index)
        
    elif strategy == 'entropy':
        
        probs = pd.DataFrame(data=np.apply_along_axis(entropy, 1, probs))
        probs.sort_values(by=probs.columns[0], ascending=True, inplace=True)
        unlabeled_data = unlabeled_data.reindex(probs.index)
        df_temp = unlabeled_data.head(sample_size)
        
        X_expanded = pd.DataFrame(data=np.concatenate([X_train.values.reshape((X_train.shape[0],1)), df_temp[train_col].values.reshape((sample_size, 1))], axis=0), columns = [train_col]) 
        y_expanded = pd.concat([y_train, df_temp[target_col]], sort=True)
        unlabeled_reduced = unlabeled_data.drop(df_temp.head(sample_size).index)       
    
    elif strategy == 'least_confident':
        
        probs = pd.DataFrame(data=(1-np.amax(probs.values, axis=1)))        
        probs.sort_values(by=probs.columns[0], ascending=True, inplace=True)
        unlabeled_data = unlabeled_data.reindex(probs.index)
        df_temp = unlabeled_data.head(sample_size)        
        X_expanded = pd.DataFrame(data=np.concatenate([X_train.values.reshape((X_train.shape[0],1)), df_temp[train_col].values.reshape((sample_size, 1))], axis=0), columns = [train_col])    
        y_expanded = pd.concat([y_train, df_temp[target_col].reset_index(drop=True)],ignore_index=True, sort=True)
        unlabeled_reduced = unlabeled_data.drop(df_temp.head(sample_size).index)

        
    else:
        print('The chosen strategy is not implemented yet.') 
        print('Implemented strategies: \n random \n entropy \n least_confident \n max_margin')

    if save:
        df_temp.to_csv(filename, index=False)
    
    return X_expanded, y_expanded, unlabeled_reduced