#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:00:21 2018

@author: kenn
"""
import pandas as pd
import sklearn as skl
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt


missing_90 = ['x_0'+str(i) for i in range(62,73+1)] + ['x_0'+str(i) for i in range(81,87+1)] + ['x_0'+str(i) for i in range(92,99+1)] + ['x_'+str(i) for i in range(100,120+1)] + ['x_'+str(i) for i in range(128,130+1)]

pos_neg_rate = 0
importance = []

num_leaves = 20
max_depth = 9
feature_fraction = 0.6
bagging_fraction = 0.5
num_trees = 150
learning_rate = 0.05
update_var_score = True
use_var_score = False
var_score_drop = 0.01




def balance_data(data):
    global pos_neg_rate 
    pos_neg_rate = 0
    dataset_1 = data.loc[data['y']==1,]
    dataset_0 = data.loc[data['y']==0,]
    data_0_split_index = np.array((list(range(int(len(dataset_0)/len(dataset_1))))*len(dataset_0))[:len(dataset_0)])
    train_split = []
    for i in set(data_0_split_index):
        temp = dataset_0[data_0_split_index==i]
        pos_neg_rate += len(temp)/(len(temp)+len(dataset_1))/len(set(data_0_split_index))
        train_split.append(dataset_0[data_0_split_index==i].append(dataset_1))
    return train_split

def lgb_splitdata_train_balanced(i, param = {'num_leaves':num_leaves,'max_depth':max_depth, 'feature_fraction':feature_fraction, 
                                             'bagging_fraction':bagging_fraction , 'num_trees':num_trees, 'learning_rate':learning_rate, 
                                             'objective':'binary','is_unbalance':True}, num_round = num_trees,):
    global importance

    train_X = i.drop(['y'],axis=1)
    train_Y = i['y']
    train_data = lgb.Dataset(train_X, label=train_Y,  categorical_feature=categorical_feature)
    
    
    bst = lgb.train(param, train_data, num_round)
    
    
    
    importance.append(pd.DataFrame(bst.feature_importance(),index=train_X.columns, columns=["name"]))
    
    return bst

def lgb_splitdata_train(i, param = {'num_leaves':num_leaves,'max_depth':max_depth, 'feature_fraction':feature_fraction, 'bagging_fraction':bagging_fraction , 
                                    'num_trees':num_trees, 'learning_rate':learning_rate, 'objective':'binary'}, num_round = num_trees,):
    global importance

    train_X = i.drop(['y'],axis=1)
    train_Y = i['y']
    train_data = lgb.Dataset(train_X, label=train_Y,  categorical_feature=categorical_feature)
    
    
    bst = lgb.train(param, train_data, num_round)
    
    
    
    importance.append(pd.DataFrame(bst.feature_importance(),index=train_X.columns, columns=["name"]))
    
    return bst



def model_train_f(split_dataset):
    
    
    model_set = []
    for i in split_dataset:
        model_set.append(lgb_splitdata_train(i))
    return model_set

def model_pred_f(data_test,model):
    test_pred = []
    for i in model:
        test_pred.append(i.predict(data_test).reshape(-1,1))
    test_pred = np.concatenate(test_pred,axis=1)
    test_pred = np.mean(test_pred,axis=1).reshape(-1,1)
    
    return test_pred
    
def cv_process(model_data, model_train_f, model_pred_f, k):
    global cv_res
    cv_res = []
    cv_index = np.array((list(range(k))*len(model_data))[:len(model_data)])
    global F1
    F1 = []
    global AUC
    AUC = []
    for cv_i in set(cv_index):
        dataset_train = model_data[cv_index!=cv_i]
        
        dataset_test = model_data[cv_index==cv_i]
        data_test_X = dataset_test.drop(['y'],axis=1)
        data_test_Y = dataset_test['y']
        
        split_dataset = balance_data(dataset_train)
        model_fit = model_train_f(split_dataset)
        test_prob = model_pred_f(data_test_X, model_fit)
        print(pos_neg_rate)
        test_pred = skl.preprocessing.binarize(test_prob,pos_neg_rate)
        
        cv_res.append(np.column_stack((data_test_Y, 
                                       test_prob.reshape(-1), 
                                       test_pred.reshape(-1))))
        

           
        F1.append(skl.metrics.f1_score(data_test_Y, test_pred))
        AUC.append(skl.metrics.roc_auc_score(data_test_Y, test_prob))
        
    return np.mean(F1), np.sqrt(np.var(F1)), np.mean(F1)-np.sqrt(np.var(F1)), np.mean(F1)+np.sqrt(np.var(F1)) ,np.mean(AUC), np.sqrt(np.var(AUC))

def cv_process_balanced(model_data, model_train_f, model_pred_f, k):
    global cv_res
    cv_res = []
    cv_index = np.array((list(range(k))*len(model_data))[:len(model_data)])
    global F1
    F1 = []
    global AUC
    AUC = []
    for cv_i in set(cv_index):
        dataset_train = model_data[cv_index!=cv_i]
        
        dataset_test = model_data[cv_index==cv_i]
        data_test_X = dataset_test.drop(['y'],axis=1)
        data_test_Y = dataset_test['y']
        
        model = lgb_splitdata_train_balanced(dataset_train)
        test_prob = model.predict(data_test_X).reshape(-1,1)
        print(pos_neg_rate)
        test_pred = skl.preprocessing.binarize(test_prob,0.5)
        
        cv_res.append(np.column_stack((data_test_Y, 
                                       test_prob.reshape(-1), 
                                       test_pred.reshape(-1))))
        

           
        F1.append(skl.metrics.f1_score(data_test_Y, test_pred))
        AUC.append(skl.metrics.roc_auc_score(data_test_Y, test_prob))
        print(np.mean(F1), np.sqrt(np.var(F1)), np.mean(F1)-np.sqrt(np.var(F1)), np.mean(F1)+np.sqrt(np.var(F1)) ,np.mean(AUC))
    return [np.mean(F1), np.sqrt(np.var(F1)), np.mean(F1)-np.sqrt(np.var(F1)), np.mean(F1)+np.sqrt(np.var(F1)) ,np.mean(AUC)]

dataset = pd.read_csv('model_sample.csv')
categorical_feature = ['x_001','x_010','x_011','x_027','x_033'] + ['x_00'+str(i) for i in range(3,9+1)] + ['x_0'+str(i) for i in range(13,19+1)]
dataset.drop(['user_id','x_012'],axis=1,inplace=True)
dataset.drop(missing_90,axis=1,inplace=True)

if use_var_score:
    var_score = pd.read_csv("var_score.csv")
    var_score.columns=['name','score']
    drop_var = list(var_score.loc[var_score['score']<=var_score_drop,'name'])
    categorical_feature = list(set(categorical_feature).difference(set(drop_var)))
    dataset.drop(drop_var,axis=1,inplace=True)

model_data = dataset


print(cv_process(model_data, model_train_f, model_pred_f, 10))

if update_var_score:
    importance_matrix = pd.concat(importance,axis=1)
    importance_matrix = (importance_matrix - np.min(importance_matrix))/(np.max(importance_matrix) - np.min(importance_matrix))
    var_score = pd.DataFrame(np.mean(importance_matrix,axis=1),columns=["score"])
    var_score.sort_values(by="score",ascending=False,inplace=True)
    var_score.to_csv("var_score.csv")



