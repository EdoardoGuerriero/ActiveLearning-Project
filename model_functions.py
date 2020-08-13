#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from temperature_scaling import ModelWithTemperature
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import random
import time
from utils import load_field, load_iterator
from ipdb import set_trace

# Multichannel CNN, Kim(2014) 
class MCNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        #input = [sent len, batch size]
        
        text = text.permute(1, 0)
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


# metrics for model evaluation
def categorical_accuracy(preds, y):
    # Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def precision(preds, y):
    # Convert tensors into numpy array to use sklearn functions
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1)
    return precision_score(y.detach().numpy(), max_preds.detach().numpy(), average='macro')

def recall(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1)
    return recall_score(y.detach().numpy(), max_preds.detach().numpy(), average='macro')

def f_score(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1)
    return f1_score(y.detach().numpy(), max_preds.detach().numpy(), average='macro')

def classes_stats(preds,y, LABEL):
    max_preds = preds.argmax(dim = 1, keepdim = True).squeeze(1)
    report = classification_report(y.detach().numpy(), max_preds.detach().numpy(), output_dict=True)
    
    # extract dictionaries from classification report
    #set_trace()
    
    stats_against = report[str(LABEL.vocab.stoi['AGAINST'])]
    stats_favor = report[str(LABEL.vocab.stoi['FAVOR'])]
    stats_neutral = report[str(LABEL.vocab.stoi['NEUTRAL'])]
    stats_micro = report['micro avg']
    stats_macro = report['macro avg']
    stats_weighted = report['weighted avg']
    
    return stats_against, stats_favor, stats_neutral, stats_micro, stats_macro, stats_weighted

# training function 
# (history is a dict e.g. train_history={'accuracy':[],'loss'=[], etc..})
def train(model, iterator, optimizer, criterion, history):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        rec = recall(predictions, batch.label)
        pre = precision(predictions, batch.label)
        f1 = f_score(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_recall += rec
        epoch_precision += pre
        epoch_f1 += f1
    
    epoch_loss = epoch_loss/len(iterator)
    epoch_acc = epoch_acc/len(iterator)
    epoch_recall = epoch_recall/len(iterator)
    epoch_precision = epoch_precision/len(iterator)
    epoch_f1 = epoch_f1/len(iterator)
    
    history['accuracy'].append(epoch_acc)
    history['loss'].append(epoch_loss)
    history['recall'].append(epoch_recall)
    history['precision'].append(epoch_precision)
    history['f1'].append(epoch_f1)
    
    return epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1 

# evaluation function 
# (history is a dict e.g. test_history={'accuracy':[],'loss'=[]})
def evaluate(model, iterator, criterion, history):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            rec = recall(predictions, batch.label)
            pre = precision(predictions, batch.label)
            f1 = f_score(predictions, batch.label)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_recall += rec
            epoch_precision += pre
            epoch_f1 += f1
            
    epoch_loss = epoch_loss/len(iterator)
    epoch_acc = epoch_acc/len(iterator)
    epoch_recall = epoch_recall/len(iterator)
    epoch_precision = epoch_precision/len(iterator)
    epoch_f1 = epoch_f1/len(iterator)
    
    history['accuracy'].append(epoch_acc)
    history['loss'].append(epoch_loss)
    history['recall'].append(epoch_recall)
    history['precision'].append(epoch_precision)
    history['f1'].append(epoch_f1)
    
    return epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1

# evaluate including more specific statistics
def evaluate_expanded(model, iterator, criterion, history, LABEL):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
        
    epoch_stats_against = {'precision':0,'recall':0,'f1-score':0,'support':0}
    epoch_stats_favor = {'precision':0,'recall':0,'f1-score':0,'support':0}
    epoch_stats_neutral = {'precision':0,'recall':0,'f1-score':0,'support':0}
    epoch_stats_macro = {'precision':0,'recall':0,'f1-score':0,'support':0}
    epoch_stats_micro = {'precision':0,'recall':0,'f1-score':0,'support':0}
    epoch_stats_weighted = {'precision':0,'recall':0,'f1-score':0,'support':0}
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            # single values
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            rec = recall(predictions, batch.label)
            pre = precision(predictions, batch.label)
            f1 = f_score(predictions, batch.label)
            
            # dictionaries 
            stats_against, stats_favor, stats_neutral, stats_micro, stats_macro, stats_weighted =\
                classes_stats(predictions, batch.label, LABEL)
            
            # update values 
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_recall += rec
            epoch_precision += pre
            epoch_f1 += f1
            
            # update dictionatries 
            epoch_stats_against = \
                { k: epoch_stats_against.get(k, 0) + stats_against.get(k, 0) \
                 for k in set(epoch_stats_against) | set(stats_against) }
            epoch_stats_favor = \
                { k: epoch_stats_favor.get(k, 0) + stats_favor.get(k, 0) \
                 for k in set(epoch_stats_favor) | set(stats_favor) }
            epoch_stats_neutral = \
                { k: epoch_stats_neutral.get(k, 0) + stats_neutral.get(k, 0) \
                 for k in set(epoch_stats_neutral) | set(stats_neutral) }
            epoch_stats_macro = \
                { k: epoch_stats_macro.get(k, 0) + stats_macro.get(k, 0) \
                 for k in set(epoch_stats_macro) | set(stats_macro) }
            epoch_stats_micro = \
                { k: epoch_stats_micro.get(k, 0) + stats_micro.get(k, 0) \
                 for k in set(epoch_stats_micro) | set(stats_micro) }
            epoch_stats_weighted = \
                { k: epoch_stats_weighted.get(k, 0) + stats_weighted.get(k, 0) \
                 for k in set(epoch_stats_weighted) | set(stats_weighted) }
    
    len_it = len(iterator)
    
    # compute average values
    epoch_loss = epoch_loss/len_it
    epoch_acc = epoch_acc/len_it
    epoch_recall = epoch_recall/len_it
    epoch_precision = epoch_precision/len_it
    epoch_f1 = epoch_f1/len_it
    
    # compute average dictionaries items
    epoch_stats_against = {k: epoch_stats_against[k]/len_it for k in epoch_stats_against.keys()}
    epoch_stats_favor = {k: epoch_stats_favor[k]/len_it for k in epoch_stats_favor.keys()}
    epoch_stats_neutral = {k: epoch_stats_neutral[k]/len_it for k in epoch_stats_neutral.keys()}
    epoch_stats_macro = {k: epoch_stats_macro[k]/len_it for k in epoch_stats_macro.keys()}
    epoch_stats_micro = {k: epoch_stats_micro[k]/len_it for k in epoch_stats_micro.keys()}
    epoch_stats_weighted = {k: epoch_stats_weighted[k]/len_it for k in epoch_stats_weighted.keys()}
    
    history['accuracy'].append(epoch_acc)
    history['loss'].append(epoch_loss)
    history['recall'].append(epoch_recall)
    history['precision'].append(epoch_precision)
    history['f1'].append(epoch_f1)
    history['against'].append(epoch_stats_against)
    history['favor'].append(epoch_stats_favor)
    history['neutral'].append(epoch_stats_neutral)
    history['macro'].append(epoch_stats_macro)
    history['weighted'].append(epoch_stats_weighted)
    
    return epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1


# function to compute the time needed to train a single epoch
def epoch_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs

# function with pipeline to predict single sentences 
def predict_class(model, sentence, nlp, TEXT, device, min_len = 4,\
                  active_learning=False,active_dropout=False):
    
    if active_dropout:
        model.train()
    else:
        model.eval()
        
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
        
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    
    if active_learning:
        preds = F.softmax(model(tensor), dim=1)
        max_preds = preds.argmax(dim = 1)
    
        return preds, max_preds.item()
    
    else:
        return model(tensor)
    
# function to calibrate models predictions (post-processing step)
def model_calibration(model, iterator):
    
    orig_model = model # create an uncalibrated model somehow
    valid_loader = iterator # Create a DataLoader from the SAME VALIDATION SET used to train orig_model
    
    scaled_model = ModelWithTemperature(orig_model)
    scaled_model.set_temperature(valid_loader)
    
    return scaled_model

def training_sesssion(datasets_path, train_name, step, model_path=None):
    
    # initialize pytorch stuff
    torch.backends.cudnn.deterministic = True
    
    TEXT = data.Field(tokenize = 'spacy')
    LABEL = data.LabelField()
        
    train_data, test_data = data.TabularDataset.splits(
            path=datasets_path, train=train_name,
            test='test_final.csv', format='csv',
            fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    
    # split train into train and validation
    train_data, valid_data = train_data.split(split_ratio=0.7,random_state=random.seed(2))
    
    # build vocabulary and define other pytorch variables
    MAX_VOCAB_SIZE = 10000
    TEXT.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = "glove.6B.50d", 
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    BATCH_SIZE = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define iterators 
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                        (train_data, valid_data, test_data), 
                                        batch_size = BATCH_SIZE, 
                                        sort_key=lambda x:len(x.text),
                                        sort_within_batch=False,
                                        device = device)
    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 50
    N_FILTERS = 50
    FILTER_SIZES = [2,3]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    # define model
    model = MCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    
    # set embedding vectors 
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    # set optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    N_EPOCHS = 5
    
    best_valid_loss = float('inf')
    
    train_history = {'accuracy':[], 'loss':[], 'recall':[], 'precision':[], 'f1':[]}
    valid_history = {'accuracy':[], 'loss':[], 'recall':[], 'precision':[], 'f1':[]}
    
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss, train_acc,_,_,_ = train(model, train_iterator, optimizer, \
                                      criterion, train_history)
        valid_loss, valid_acc,_,_,_ = evaluate(model, valid_iterator, \
                                         criterion, valid_history)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
    
    test_history = {'accuracy':[], 'loss':[], 'recall':[], 'precision':[], 'f1':[]}
    test_loss, test_acc,_,_,_ = evaluate(model, test_iterator, criterion, test_history)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    # model calibration
    scaled_model = model_calibration(model, valid_iterator)
    
    return model, scaled_model, train_history, valid_history, test_history, TEXT, LABEL, device
    


def retraining_session(datasets_path, path_TEXT, train_name, step, model_path, sim, strategy):
    
    # initialize pytorch stuff
    torch.backends.cudnn.deterministic = True
    
    if strategy != 'random':
        TEXT = load_field('Random 50bi-tri calib/Random '+sim+'/Step_1/TEXT')
        LABEL = load_field('Random 50bi-tri calib/Random '+sim+'/Step_1/LABEL')
    else:
        TEXT = load_field(path_TEXT+'TEXT')
        LABEL = load_field(path_TEXT+'LABEL')
    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 50
    N_FILTERS = 50
    FILTER_SIZES = [2,3]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    # load previous parameters for simulations after the first one
    model = MCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    model.load_state_dict(torch.load(model_path))
    
    train_data, test_data = data.TabularDataset.splits(
        path=datasets_path, train=train_name,
        test='test_final.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    
    # split train into train and validation
    train_data, valid_data = train_data.split(split_ratio=0.7,random_state=random.seed(2))
    
    BATCH_SIZE = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define iterators 
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                        (train_data, valid_data, test_data), 
                                        batch_size = BATCH_SIZE, 
                                        sort_key=lambda x:len(x.text),
                                        sort_within_batch=False,
                                        device = device)
    
    # set optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    N_EPOCHS = 5
    
    best_valid_loss = float('inf')
    
    train_history = {'accuracy':[], 'loss':[], 'recall':[], 'precision':[], 'f1':[]}
    valid_history = {'accuracy':[], 'loss':[], 'recall':[], 'precision':[], 'f1':[]}
    
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss, train_acc,_,_,_ = train(model, train_iterator, optimizer, \
                                      criterion, train_history)
        valid_loss, valid_acc,_,_,_ = evaluate(model, valid_iterator, \
                                         criterion, valid_history)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        
    test_history = {'accuracy':[], 'loss':[], 'recall':[], 'precision':[], 'f1':[]}
    test_loss, test_acc,_,_,_ = evaluate(model, test_iterator, criterion, test_history)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    # model calibration
    scaled_model = model_calibration(model, valid_iterator)
    
    return model, scaled_model, train_history, valid_history, test_history, TEXT, LABEL, device


def load_model(sim):
    
    TEXT = load_field('Random 50bi-tri calib/Random '+sim+'/Step_1/TEXT')
    LABEL = load_field('Random 50bi-tri calib/Random '+sim+'/Step_1/LABEL')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # need to re-define here all these parameters here as well
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 50
    N_FILTERS = 50
    FILTER_SIZES = [2,3]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # load previous parameters for simulations after the first one
    model = MCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    model.load_state_dict(torch.load('Random 50bi-tri calib/Random '+sim+'/Step_1/model.pt'))
    
    return model, TEXT, LABEL, device
        



