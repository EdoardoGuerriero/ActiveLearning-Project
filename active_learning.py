#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import entropy
from model_functions import predict_class 
import spacy 
from ipdb import set_trace

def active_step(unlabelled_data, model, step, TEXT, device, \
                strategy='entropy', sample_size=50):
    
    nlp = spacy.load('en')
    
    if (strategy == 'entropy') or (strategy == 'cost-effective'):
        
        # calculate entropy for each instance in unlabelled data
        for ind, row in unlabelled_data.iterrows():
            preds = predict_class(model, row['text'],nlp, TEXT, \
                                  device, active_learning=True,)
            unlabelled_data.loc[ind,'entropy'] = entropy(preds[0].detach().numpy()[0])
        
        unlabelled_data.sort_values(by='entropy',axis=0,ascending=False,inplace=True)
        
        # return only low confidence data indices 
        if strategy == 'entropy':
            low_conf_indices = unlabelled_data.head(sample_size).index
            return low_conf_indices 
        
        # return low confidence and high confidence data indices
        # (except for last step when unlabelled size = sample size)
        else:
            
            low_conf_indices = unlabelled_data.head(sample_size).index
            
            if step == 16:
                return low_conf_indices
                
            else:
                # threshold = 0.5
                high_conf_indices = unlabelled_data.tail(10).index
                return low_conf_indices, high_conf_indices
    
    elif strategy == 'random':
        
        random_indices = np.random.choice(unlabelled_data.index,size=sample_size)
        #random_indices = unlabelled_data.sample(n=sample_size).index
        return random_indices
    
    elif strategy == 'max-margin':
        
        for ind, row in unlabelled_data.iterrows():
            preds = predict_class(model, row['text'],nlp, TEXT, \
                                  device, active_learning=True,)
            unlabelled_data.loc[ind,'margin'] = \
                max(preds[0].detach().numpy()[0]) - min(preds[0].detach().numpy()[0])
                #max(preds[0].detach().numpy()[0]) - np.sort(preds[0].detach().numpy()[0])[-2]
        
        unlabelled_data.sort_values(by='margin',axis=0,ascending=True,inplace=True)
        low_conf_indices = unlabelled_data.head(sample_size).index
        
        return low_conf_indices 
    
    elif strategy == 'least-conf':
        
        for ind, row in unlabelled_data.iterrows():
            preds = predict_class(model, row['text'],nlp, TEXT, \
                                  device, active_learning=True,)
            unlabelled_data.loc[ind,'confidence'] = \
                max(preds[0].detach().numpy()[0]) 
        
        unlabelled_data.sort_values(by='confidence',axis=0,ascending=True,inplace=True)
        low_conf_indices = unlabelled_data.head(sample_size).index
        
        return low_conf_indices 
        
    elif strategy == 'monte-carlo':
        
        # arrays to store logits and entropy scores
        # score_all --> entropy(mean(logits))
        # entropy_all --> mean(entropy(logits))
        score_all = np.zeros((unlabelled_data.shape[0]))
        entropy_all = np.zeros(unlabelled_data.shape[0])
        
        # active_dropout=True to set model in training mode 
        for ind, row in unlabelled_data.iterrows():
            
            # store 100 different predictions
            temp_preds = np.zeros((100,3))
            entropy_row_score = 0
            
            for mc in range(100):
                preds = predict_class(model, row['text'],nlp, TEXT, device, \
                                      active_learning =True, active_dropout=True)
                temp_preds[:][mc] = preds[0].detach().numpy()[0]
                entropy_row_score += entropy(preds[0].detach().numpy()[0])**2
            
            # update arrays
            score_all_row = np.sum(temp_preds,axis=0)/100
            score_all[:][ind] = entropy(score_all_row)
            entropy_all[ind] = entropy_row_score
        
        # compute variance 
        G_X = score_all
        F_X = (entropy_all/100)**2
        U_X = G_X - F_X
        
        unlabelled_data['variance'] = U_X
        unlabelled_data.sort_values(by='variance',axis=0,ascending=False,inplace=True)
        low_conf_indices = unlabelled_data.head(sample_size).index
        
        model.eval()
        
        return low_conf_indices
                

    
    
def active_step_shallow(unlabelled_df, model, vect, col, \
                strategy='entropy', sample_size=50):
    
    unlabelled_data = unlabelled_df.copy()
    
    if strategy != 'random':
        for ind, row in unlabelled_data.iterrows():
            
            row_text = [row[col]]
            text = vect.transform(row_text)
            probs = model.predict_proba(text.toarray())[0]
            
            unlabelled_data.loc[ind,'least_conf'] = max(probs)
#            set_trace()
            unlabelled_data.loc[ind,'max_margin'] = max(probs) - np.sort(probs)[-2]
            unlabelled_data.loc[ind,'entropy'] = entropy(probs)
            
        # return only low confidence data indices 
        if strategy == 'entropy':
            unlabelled_data.sort_values(by='entropy',axis=0,ascending=False,inplace=True)
            low_conf_indices = unlabelled_data.head(sample_size).index
        
        if strategy == 'least_conf':
            unlabelled_data.sort_values(by='least_conf',axis=0,ascending=True,inplace=True)
            low_conf_indices = unlabelled_data.head(sample_size).index
            
        if strategy == 'max_margin':
            unlabelled_data.sort_values(by='max_margin',axis=0,ascending=True,inplace=True)
            low_conf_indices = unlabelled_data.head(sample_size).index
        
        return low_conf_indices
    
    else:
        low_conf_indices = unlabelled_data.sample(n=sample_size).index
        
        return low_conf_indices