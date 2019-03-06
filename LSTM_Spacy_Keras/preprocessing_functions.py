#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:05:43 2019

@author: edoardoguerriero
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import re 
from urllib.parse import urlparse


# remove tags, hashtags simbol, url
def cleaning_text(data, text_col):
    first_cleaned_text = []
    number_hashtags = []
    list_hashtags = []
    
    for t, tweet in data.iterrows():
        text = tweet[text_col]
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
def rename_labels(data, target_col):
    
    labels = data[target_col].unique()
    
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
def split_save(test_train_data, unlabeled_data, test_size, step, text_col, target_col, path_step):
    
    text = test_train_data['first_cleaning_text']
    stance = test_train_data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(text, stance, test_size=test_size)
    
    unlabeled_data = unlabeled_data['first_cleaning_text'].values
    
    train_dataset = pd.concat([X_train.to_frame(name=text_col).reset_index(drop=True)\
                               , y_train.to_frame(name=target_col).reset_index(drop=True)],axis=1)
    test_dataset = pd.concat([X_test.to_frame(name=text_col).reset_index(drop=True)\
                              , y_test.to_frame(name=target_col).reset_index(drop=True)],axis=1)
    # save for next step
    train_dataset.to_csv(path_step+'/train_data.csv'.format(step),index=False)
    test_dataset.to_csv(path_step+'/test_data.csv'.format(step),index=False)
    
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    
    return X_train, y_train, X_test, y_test, unlabeled_data


