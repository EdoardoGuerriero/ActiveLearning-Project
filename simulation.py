#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:57:06 2019

@author: edoardoguerriero
"""
import os 
import pandas as pd
import torch
from model_functions import training_sesssion, retraining_session, load_model
from utils import save_obj, save_field, save_iterator
from active_learning import active_step

def train_with_active_learning(folder_name, strategy, sim, step=1, max_step=9):
    
    calibration = True
    
    # check if target folder exist
    path_target = './{}'.format(folder_name)
    if not os.path.exists(path_target):
        os.makedirs(path_target)
    
    # name of subfolder (one will be created for each step)
    origin_datasets_path = 'final datasets/'
    path_step = path_target + '/Step_{}'.format(step)
    path_prev_step = path_target + '/Step_{}'.format(step-1)
    path_TEXT = path_target + '/Step_1/'
    if not os.path.exists(path_step):
        os.makedirs(path_step)
    
    # training for first step
    if step == 1:
        
        # path original datasets
        train_name = 'train_validation_final.csv'
        
        # read training and test datasets
        train_df = pd.read_csv(origin_datasets_path+'train_validation_final.csv')
        test_df = pd.read_csv(origin_datasets_path+'test_final.csv')
#        unlabeled_data = pd.read_csv(origin_datasets_path+'active_selection_final.csv')
        unlabeled_data = pd.read_csv(origin_datasets_path+'Edo_400_final.csv')
        
        if strategy == 'random':
            
            # train model
            model, scaled_model, train_history, valid_history, test_history, TEXT, LABEL, device = \
                            training_sesssion(origin_datasets_path, train_name, step)
            print('\nDone training')
            
            # save model, fields, history dicts
            torch.save(model.state_dict(), path_step+'/model.pt')
            torch.save(scaled_model.state_dict(), path_step+'/scaled_model.pt')
            save_obj(train_history, path_step+'/train_history')
            save_obj(valid_history, path_step+'/valid_history')
            save_obj(test_history, path_step+'/test_history')
            #save_iterator(valid_iterator, path_step+'/valid_iterator')
            save_field(TEXT, path_step+'/TEXT')
            save_field(LABEL, path_step+'/LABEL')
            print('\nDone saving model and stats')
        
        else:
            model, TEXT, LABEL, device = load_model(sim)
        
        al_model = model
        
        # active learning step
        if strategy != 'cost-effective':
            indices_low_conf = active_step(unlabeled_data, al_model, step,\
                                           TEXT, device, strategy)
            print('\nDone active learning selection')
        else:
            indices_low_conf, indices_high_conf = active_step(unlabeled_data, \
                                            al_model, step, TEXT, device, strategy)
            print('\nDone active learning selection')
        
        # save intermediate datasets
        if strategy != 'cost-effective':
            data_to_label = unlabeled_data.loc[indices_low_conf].reset_index(drop=True)
            data_to_label.to_csv(path_step+'/tweet_low_conf.csv'.format(step), index=False)
            data_unlabeled_next_step = unlabeled_data.drop(indices_low_conf).reset_index(drop=True)
            data_unlabeled_next_step.to_csv(path_step+'/unlabeled_next_step.csv'.format(step), index=False)
            
            if strategy == 'entropy':
                augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='entropy')]).reset_index(drop=True)
            elif strategy == 'monte-carlo':
                augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='variance')]).reset_index(drop=True)
            elif strategy == 'max-margin':
                augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='margin')]).reset_index(drop=True)
            elif strategy == 'least-conf':
                augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='confidence')]).reset_index(drop=True)
            else:
                augmented_data_train = pd.concat([train_df,data_to_label]).reset_index(drop=True)
                
            augmented_data_train.to_csv(path_step+'/train_next_step.csv'.format(step), index=False)
            test_df.to_csv(path_step+'/test_final.csv'.format(step), index=False)
            print('\nDone saving datasets')
            
        else:
            data_low_conf = unlabeled_data.loc[indices_low_conf].reset_index(drop=True)
            data_low_conf.to_csv(path_step+'/tweet_low_conf.csv'.format(step), index=False)
            data_high_conf = unlabeled_data.loc[indices_low_conf].reset_index(drop=True)
            data_high_conf.to_csv(path_step+'/tweet_high_conf.csv'.format(step), index=False)
            data_unlabeled_next_step = unlabeled_data.drop(indices_low_conf).reset_index(drop=True)
            data_unlabeled_next_step.to_csv(path_step+'/unlabeled_next_step.csv'.format(step), index=False)
            try:
                augmented_data_train = pd.concat([train_df,data_low_conf.drop(columns='entropy'),\
                                                  data_high_conf.drop(columns='entropy')]).reset_index(drop=True)
            except:
                augmented_data_train = pd.concat([train_df,data_low_conf.drop(columns='variance'),\
                                                  data_high_conf.drop(columns='variance')]).reset_index(drop=True)
            augmented_data_train.to_csv(path_step+'/train_next_step.csv'.format(step), index=False)
            test_df.to_csv(path_step+'/test_final.csv'.format(step), index=False)
            print('\nDone saving datasets')
    
    # loop for step after the first one
    else:
        # path model / datasets
        train_name =  'train_next_step.csv' #'tweet_low_conf.csv' 
        
        if (step==2) and (strategy!='random'):
            prev_model = 'Random 50bi-tri calib/Random '+sim+'/Step_1/model.pt'
        else:
            prev_model = path_prev_step+'/model.pt'
        
        # load datasets from previous step
        try:
            train_df = pd.read_csv(path_prev_step+'/'+train_name)
            train_df = train_df.sample(frac=1).reset_index(drop=True)
            test_df = pd.read_csv(origin_datasets_path+'test_final.csv')
            unlabeled_data = pd.read_csv(path_prev_step+'/unlabeled_next_step.csv')
        
        except:
            print("\nCan't load training files from previous step folder.")
        
        # train model
        model, scaled_model, train_history, valid_history, test_history, TEXT, LABEL, device = \
                retraining_session(path_prev_step+'/', path_TEXT, train_name, step, prev_model,sim, strategy)
        
        print('\nDone training')
        
        # save model, fields, history dicts
        torch.save(model.state_dict(), path_step+'/model.pt')
        torch.save(scaled_model.state_dict(), path_step+'/scaled_model.pt')
        save_obj(train_history, path_step+'/train_history')
        save_obj(valid_history, path_step+'/valid_history')
        save_obj(test_history, path_step+'/test_history')
        print('\nDone saving model and stats')
        
        if calibration:
            al_model = scaled_model
        else:
            al_model = model
        
        # active learning step
        if step != max_step:
                
            if (strategy != 'cost-effective') or (strategy == 'cost-effective' and step==16):
                indices_low_conf = active_step(unlabeled_data, al_model, step,\
                                               TEXT, device, strategy)
                print('\nDone active learning selection')
                
            else:
                indices_low_conf, indices_high_conf = active_step(unlabeled_data, \
                                                al_model, step, TEXT, device, strategy)
                print('\nDone active learning selection')

            if (strategy != 'cost-effective') or (strategy=='cost-effective' and step==16):
                data_to_label = unlabeled_data.loc[indices_low_conf].reset_index(drop=True)
                data_to_label.to_csv(path_step+'/tweet_low_conf.csv'.format(step), index=False)
                data_unlabeled_next_step = unlabeled_data.drop(indices_low_conf).reset_index(drop=True)
                data_unlabeled_next_step.to_csv(path_step+'/unlabeled_next_step.csv'.format(step), index=False)
                
                if strategy == 'entropy' or strategy == 'cost-effective':
                    augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='entropy')]).reset_index(drop=True)
                elif strategy == 'monte-carlo':
                    augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='variance')]).reset_index(drop=True)
                elif strategy == 'max-margin':
                    augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='margin')]).reset_index(drop=True)
                elif strategy == 'least-conf':
                    augmented_data_train = pd.concat([train_df,data_to_label.drop(columns='confidence')]).reset_index(drop=True)
                else:
                    augmented_data_train = pd.concat([train_df,data_to_label]).reset_index(drop=True)
                    
                augmented_data_train.to_csv(path_step+'/train_next_step.csv'.format(step), index=False)
                test_df.to_csv(path_step+'/test_final.csv'.format(step), index=False)
                print('\nDone saving datasets')
                
            else:
                data_low_conf = unlabeled_data.loc[indices_low_conf].reset_index(drop=True)
                data_low_conf.to_csv(path_step+'/tweet_low_conf.csv'.format(step), index=False)
                data_high_conf = unlabeled_data.loc[indices_low_conf].reset_index(drop=True)
                data_high_conf.to_csv(path_step+'/tweet_high_conf.csv'.format(step), index=False)
                data_unlabeled_next_step = unlabeled_data.drop(indices_low_conf).reset_index(drop=True)
                data_unlabeled_next_step.to_csv(path_step+'/unlabeled_next_step.csv'.format(step), index=False)
                try:
                    augmented_data_train = pd.concat([train_df,data_low_conf.drop(columns='entropy'),\
                                                      data_high_conf.drop(columns='entropy')]).reset_index(drop=True)
                except:
                    augmented_data_train = pd.concat([train_df,data_low_conf.drop(columns='variance'),\
                                                      data_high_conf.drop(columns='variance')]).reset_index(drop=True)
                augmented_data_train.to_csv(path_step+'/train_next_step.csv'.format(step), index=False)
                test_df.to_csv(path_step+'/test_final.csv'.format(step), index=False)
                print('\nDone saving datasets')
        
    # remove uneuseful variables to save memory
    # gc.collect()
    
