#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 11:11:12 2018

@author: kenn
"""


import pandas as pd
import sklearn as skl
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
import copy
from sklearn.preprocessing import OneHotEncoder


missing_90 = ['x_0'+str(i) for i in range(62,73+1)] + ['x_0'+str(i) for i in range(81,87+1)] + ['x_0'+str(i) for i in range(92,99+1)] + ['x_'+str(i) for i in range(100,120+1)] + ['x_'+str(i) for i in range(128,130+1)]
month_var = ['x_042', 'x_049', 'x_056', 'x_075', 'x_089', 'x_122']
raw01_var = ['x_00'+str(i) for i in range(3,9+1)] + ['x_0'+str(i) for i in range(10,19+1)] + ['x_001']

cv_i=0
k=5

num_leaves = 4 #woe 3 onehot 4
max_depth = 9
feature_fraction = 0.6
bagging_fraction = 1 #useless
num_trees = 200
learning_rate = 0.05
iv_threshold = 0.2
use_handcraft_feature = True
bin_after_split = False
import_threshod = 0.1
use_lgb_feature = True
lgb_feature_iv_screen = True
raw_feature_iv_screen = True
use_woe_feature = False
use_dummy = False
use_l2 = True


var_score = pd.read_csv("var_score.csv")
var_score.columns = ["name","score"]
drop_var = var_score.loc[var_score["score"]<import_threshod,["name"]]



pos_neg_rate = 0
max_tree_tranform = num_trees
transform_matrix_name = ["tree_"+str(j)+"_leave_"+str(i) for j in range(max_tree_tranform) for i in range(num_leaves)  ]




def _read_tree_split_value(tree_structure, split_point_list):
    if isinstance(tree_structure, dict):
        for k, v in tree_structure.items():
            if k == 'threshold':
                split_point_list.append(v)
            if isinstance(v, dict):
                _read_tree_split_value(v, split_point_list)


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


def lgb_splitdata_train(x, param = {'num_leaves':num_leaves,'max_depth':max_depth, 
                                    'feature_fraction':feature_fraction, 'bagging_fraction':bagging_fraction , 
                                    'num_trees':num_trees, 'learning_rate':learning_rate, 
                                    'objective':'binary', "is_unbalance":True}, num_round = num_trees,):
    
    cf = list(set(categorical_feature).intersection(set(x.columns)))
    train_X = x.drop(['user_id','y'],axis=1)
    train_Y = x['y']
    train_data = lgb.Dataset(train_X, label=train_Y,  categorical_feature=cf)
    
    bst = lgb.train(param, train_data, num_round)
    return bst

def lr_splitdata_train(x, lr=None):
    
    if lr is None:
        if use_l2:
            lr = linear_model.LogisticRegressionCV(Cs = 10, penalty='l2', solver="liblinear", cv = 5, class_weight='balanced', n_jobs=-1)
        else:
            lr = linear_model.LogisticRegressionCV(Cs = 10, penalty='l1', solver="liblinear", cv = 5, class_weight='balanced', n_jobs=-1)
    train_X = x.drop(['user_id','y'],axis=1)
    train_Y = x['y']
    
    lr.fit(train_X, train_Y)

    return lr

def get_leaf_features(bst, x):
    pred = bst.predict(x.drop(["user_id"],axis=1), pred_leaf=True)[:,:max_tree_tranform]
    
    transform_matrix = np.zeros([len(pred),max_tree_tranform * num_leaves], dtype = np.int64)
    for i in range(0,len(pred)):
        temp= np.arange(max_tree_tranform) * num_leaves + np.array(pred[i])
        transform_matrix[i][temp] += 1
    
    transform_matrix = pd.DataFrame(np.column_stack((x["user_id"], transform_matrix)), columns = ["user_id"] + transform_matrix_name)
    return transform_matrix


def lgb_model_train(split_dataset):
    
    
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


def model_train(lgb_data, lr_data, lgb_test, lr_test):
    model = {}
        
    lgb_model = lgb_splitdata_train(lgb_data)

    model["lgb_model"] = lgb_model
    if use_lgb_feature:
        if lgb_feature_iv_screen:
            
            train_tree_node_pred = pd.DataFrame(np.column_stack((lgb_data['y'],
                                                          lgb_model.predict(lgb_data.drop(['y'],axis=1).drop(["user_id"],axis=1), pred_leaf=True)[:,:max_tree_tranform])),
                                          columns = ['y']+["tree_"+str(n) for n in range(max_tree_tranform)])
    
    
            train_tree_iv = pd.DataFrame(data_iv(train_tree_node_pred, woe_vars=["tree_"+str(n) for n in range(max_tree_tranform)]),columns=['name','iv'])
            global tree_iv_low_name
            tree_iv_low_name = list(train_tree_iv.loc[train_tree_iv['iv']<iv_threshold,'name'])
            
            train_tree_node_pred.drop(tree_iv_low_name,axis=1,inplace=True)
            
            test_tree_node_pred = pd.DataFrame(model["lgb_model"].predict(lgb_test.drop(['y'],axis=1).drop(["user_id"],axis=1), pred_leaf=True)[:,:max_tree_tranform],
                                          columns = ["tree_"+str(n) for n in range(max_tree_tranform)])
            test_tree_node_pred.drop(tree_iv_low_name,axis=1,inplace=True)
            
            if use_woe_feature:
                woe_vars = list(test_tree_node_pred.columns)
                train_tree_node, test_tree_node = data_woe(train_tree_node_pred, test_tree_node_pred, woe_vars)
                
                train_transform_matrix = pd.DataFrame(np.column_stack((lgb_data['user_id'], train_tree_node.drop(['y'],axis=1))), 
                                                      columns = ['user_id'] + list(train_tree_node.drop(['y'],axis=1).columns))
                test_transform_matrix = pd.DataFrame(np.column_stack((lgb_test['user_id'], test_tree_node)), 
                                                      columns = ['user_id'] + list(test_tree_node.columns))
            else:
                train_tree_node_pred.drop(['y'],axis=1,inplace=True)
                if use_dummy:
                    train_transform_matrix, test_transform_matrix = lr_data_onehot(train_tree_node_pred, test_tree_node_pred)
                    
                    train_transform_matrix = pd.DataFrame(np.column_stack((lgb_data["user_id"], train_transform_matrix)),
                                                          columns = ['user_id'] + list(train_transform_matrix.columns))
                    test_transform_matrix = pd.DataFrame(np.column_stack((lgb_test["user_id"], test_transform_matrix)),
                                                          columns = ['user_id'] + list(test_transform_matrix.columns))
                    
                else:
                    train_transform_matrix = np.zeros([len(lgb_data),len(train_tree_node_pred.columns) * num_leaves], dtype = np.int64)
                    for i in range(0,len(lgb_data)):
                        temp= np.arange(len(train_tree_node_pred.columns)) * num_leaves + np.array(train_tree_node_pred.iloc[i,:])
                        train_transform_matrix[i][temp] += 1
                    train_transform_matrix = pd.DataFrame(np.column_stack((lgb_data["user_id"], train_transform_matrix)), columns = ["user_id"] + [j+"_leave_"+str(i) for j in list(train_tree_node_pred.columns) for i in range(num_leaves)  ] )

                    
                    test_transform_matrix = np.zeros([len(lgb_test),len(test_tree_node_pred.columns) * num_leaves], dtype = np.int64)                
                    for i in range(0,len(lgb_test)):
                        temp= np.arange(len(test_tree_node_pred.columns)) * num_leaves + np.array(test_tree_node_pred.iloc[i,:])
                        test_transform_matrix[i][temp] += 1
                
                    test_transform_matrix = pd.DataFrame(np.column_stack((lgb_test["user_id"], test_transform_matrix)), columns = ["user_id"] + [j+"_leave_"+str(i) for j in list(test_tree_node_pred.columns) for i in range(num_leaves)  ] )
                    
                    
                
        else:
            train_transform_matrix = get_leaf_features(lgb_model, lgb_data.drop(['y'],axis=1))
            test_transform_matrix = get_leaf_features(model["lgb_model"], lgb_test.drop(["y"],axis=1))
    
        train_lr_lgb_data = lr_data.merge(train_transform_matrix, how="inner",on="user_id",validate="one_to_one")
        test_lr_lgb_data = lr_test.merge(test_transform_matrix, how="inner",on="user_id",validate="one_to_one")
    else:
        train_lr_lgb_data = lr_data
        test_lr_lgb_data = lr_test

    
    temp = train_lr_lgb_data.loc[np.sum(np.isinf(train_lr_lgb_data.drop(['user_id'],axis=1).astype(np.float32)),axis=1)==0]
    
    if len(temp) != len(train_lr_lgb_data):
        print('len temp = ',len(temp),"len train_lr_lgb_data = ",len(train_lr_lgb_data))
    
    lr_model = lr_splitdata_train(temp)

    model["lr_model"] = lr_model
    
    
    
    ##############################################################
    temp_test = test_lr_lgb_data.loc[np.sum(np.isinf(test_lr_lgb_data.drop(['user_id'],axis=1).astype(np.float32)),axis=1)==0]
    
    if len(temp_test) != len(test_lr_lgb_data):
        print('len temp_test = ',len(temp_test),"len test_lr_lgb_data = ",len(test_lr_lgb_data))
    
    pred = model["lr_model"].predict_proba(temp_test.drop(['user_id','y'],axis=1))[:,1]
    pred = pd.DataFrame(np.column_stack((temp_test['user_id'],pred)),columns=['user_id','pred'])

    return pred


def get_dt_bin(temp,var):
    min_samples_leaf=max(int(0.1*len(temp)),300)
    tree_fit = tree.DecisionTreeClassifier( class_weight="balanced",criterion="gini",min_samples_leaf=min_samples_leaf)
    tree_fit.fit(pd.DataFrame(temp[var]), temp['y']) 
    bins = [-np.inf] + sorted(np.unique([tree_fit.tree_.threshold[tree_fit.tree_.feature > -2]])) + [np.inf]
    return bins

def get_tree_bin(temp,var):
             
    tree_model = lgb.train({'num_leaves':32,'max_depth':9, 'num_trees':1, 
                   'objective':'binary',
                   'min_data_in_leaf':max(int(0.1*len(temp)),100)
                   }, lgb.Dataset(pd.DataFrame(temp[var]), label=temp['y']))
    
    tree_info = tree_model.dump_model(1)["tree_info"]
    tree_structure = tree_info[0]["tree_structure"]
    split_point_list = list()
    
    def _read_tree_split_value(tree_structure, split_point_list):
        if isinstance(tree_structure, dict):
            for k, v in tree_structure.items():
                if k == 'threshold':
                    split_point_list.append(v)
                if isinstance(v, dict):
                    _read_tree_split_value(v, split_point_list)
                    
    _read_tree_split_value(tree_structure, split_point_list)
    
    bins = [-np.inf] + sorted(split_point_list) + [np.inf]
    
    return bins

def lr_data_clean(dataset):
    data = dataset.drop(['x_012'],axis=1)
    data.loc[data['x_020']>9, 'x_020'] = 9
    data.loc[data['x_021']>9, 'x_021'] = 9
    #data.loc[data['x_022']>0, 'x_022'] = 1
    #data.loc[data['x_023']>0, 'x_023'] = 1
    data.loc[data['x_024']>3, 'x_024'] = 3
    data.loc[data['x_025']>5, 'x_025'] = 5
    data.loc[data['x_026']>6, 'x_026'] = 6
    #data.loc[data['x_027']>3, 'x_027'] = 3
    #data.loc[data['x_028']>0, 'x_028'] = 1
    #data.loc[data['x_029']>3, 'x_029'] = 3
    data.loc[data['x_030']>5, 'x_030'] = 5
    data.loc[data['x_031']>5, 'x_031'] = 5
    #data.loc[data['x_032']>2, 'x_032'] = 2
    #data.loc[data['x_033']>3, 'x_033'] = 3
    data.loc[data['x_034']>10, 'x_034'] = 10
    data.loc[np.logical_and(data['x_035']>=9,data['x_035']<=10), 'x_035'] = 9
    data.loc[data['x_035']>10, 'x_035'] = 10
    data.loc[data['x_036']>2, 'x_036'] = 2
    #data.loc[data['x_037']>0, 'x_037'] = 1
    #data.loc[data['x_038']>0, 'x_038'] = 1
    #data.loc[data['x_039']>0, 'x_039'] = 1
    #data.loc[data['x_040']>0, 'x_040'] = 1
    data.loc[data['x_089']>5, 'x_089'] = 5
    data = data.drop(missing_90,axis=1)
    data = data.drop(list(drop_var["name"]),axis=1)
    
    return data

def lr_var_split(dataset_train, dataset_test):
    
    global binset
    binset = []
    
    data_train = copy.deepcopy(dataset_train)
    data_test = copy.deepcopy(dataset_test)
    
    if use_handcraft_feature :
        cut_var = list(set(data_train.columns).difference(set(
                ['user_id','y'] + 
                month_var + 
                raw01_var + 
                missing_90 + 
                ['x_012','x_089'] + ['x_0'+str(i) for i in range(20,40+1)]))) #handcraft
    else:
        cut_var = list(set(data_train.columns).difference(set(['user_id','y'] + raw01_var)))
    
    
    for var in cut_var:
        temp_train = data_train[["y",var]].dropna()
        temp_test = data_test[["y",var]].dropna()
        bins = get_dt_bin(temp_train,var)
        binset.append(  [var , ",".join([ str(i) for i in bins])] )
        label = list(range(len(bins)-1))
        
        data_train[var] = pd.DataFrame(pd.cut(temp_train[var], bins,labels = label))
        data_test[var] = pd.DataFrame(pd.cut(temp_test[var], bins,labels = label))
    
    binset = pd.DataFrame(binset)
    binset.columns = ["name","bin"]
    binset.to_csv("binset.csv")
    
    max_label = max((max(np.max(data_train.drop(['user_id'],axis=1))) , max(np.max(data_train.drop(['user_id'],axis=1)))))
    data_train.fillna(10*(max_label+1), inplace = True)
    data_test.fillna(10*(max_label+1), inplace = True)
    
    return (data_train, data_test)


def lr_data_onehot(data_train, data_test):
    
    train = copy.deepcopy(data_train)
    test = copy.deepcopy(data_test)
    onehot_var = list(set(train.columns).difference(set(raw01_var + ['y','user_id'])))
    undo_var = list(set(train.columns).intersection(set(raw01_var + ['y','user_id'])))
    
    
    data_full = pd.concat([train[onehot_var],test[onehot_var]],axis = 0)
    
    if use_dummy:
        temp = pd.get_dummies(data_full,columns = data_full.columns,drop_first=True)
        train_X_onehot = temp.iloc[:len(train),]
        train_onehot = pd.DataFrame(np.column_stack((train[undo_var],train_X_onehot)),
                                    columns = undo_var + list(train_X_onehot.columns))
        
        test_X_onehot = temp.iloc[len(train):,]
        test_onehot = pd.DataFrame(np.column_stack((test[undo_var],test_X_onehot)),
                                   columns = undo_var + list(test_X_onehot.columns))
        
    else:
        name_list = [[i+'_'+str(j) for j in range(len(data_full[i].value_counts()))] for i in data_full.columns]
        name_list = [y for x in name_list for y in x]
        
        enc = OneHotEncoder(sparse=False)
        enc.fit( data_full )
        
        train_X_onehot = enc.transform(train[onehot_var])
        train_onehot = pd.DataFrame(np.column_stack((train[undo_var],train_X_onehot)))
        train_onehot.columns = undo_var + name_list
        
        test_X_onehot = enc.transform(test[onehot_var])
        test_onehot = pd.DataFrame(np.column_stack((test[undo_var],test_X_onehot)))
        test_onehot.columns = undo_var + name_list
    
    return (train_onehot, test_onehot)

def data_iv(data_train, woe_vars):
    train = copy.deepcopy(data_train)
    
    var_iv = []
    
    Y = train["y"]
    response=Y.sum()
    unresponse=Y.count()-response
    
    woe_var = "tree_57"
    for woe_var in woe_vars:
        d1 = pd.DataFrame({"X":train[woe_var], "Y": train["y"], "Bucket": train[woe_var]}) 
       
        
        d2 = d1.groupby('Bucket', as_index = True)        
        d3 = pd.DataFrame(d2.mean().Y)
        d3.columns = ["rate"]
        d3['sum'] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3['rate'] = d2.mean().Y
        d3['respone_distribution'] = d3['sum']/response
        d3['unresponse_distribution'] = (d3['total']-d3['sum'])/unresponse
        d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(response/unresponse))
        d3['woe2'] = np.log(d3['respone_distribution']/d3['unresponse_distribution'])
        
        d3['weight'] = d3['respone_distribution'] - d3['unresponse_distribution']
        
        d4 = d3.loc[~np.isinf(d3['woe'])]
        iv = sum(d4['woe']*d4['weight'])
        #d3.iloc[4,3]=np.inf
        
        #inf_rate = d3[np.isinf(d3['woe'])]
        
        #if len(inf_rate)>0:
            #print(woe_var)
            #print(inf_rate)
        
        var_iv.append([woe_var, iv])
    return var_iv

def data_woe(data_train, data_test, woe_vars):
    train = copy.deepcopy(data_train)
    test = copy.deepcopy(data_test)

    
    
    Y = train["y"]
    good=Y.sum()
    bad=Y.count()-good
    
    woe_var = "tree_0"
    for woe_var in woe_vars:
        d1 = pd.DataFrame({"X":train[woe_var], "Y": train["y"], "Bucket": train[woe_var]}) 
        d1_test = pd.DataFrame({"X":test[woe_var],  "Bucket": test[woe_var]}) 
        
        d2 = d1.groupby('Bucket', as_index = True)        
        d3 = pd.DataFrame(d2.mean().Y)
        d3.columns = ["rate"]
        d3['sum'] = d2.sum().Y
        d3['total'] = d2.count().Y
        d3['rate'] = d2.mean().Y
        d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
        
        #inf_rate = d3[np.isinf(d3['woe'])]
        
        #if len(inf_rate)>0:
            #print(woe_var,woe_var,woe_var,woe_var,woe_var,woe_var)
            #print(inf_rate)
            
            
        #    rule[woe_var] = inf_rate
        
        #if len(inf_rate)>1:
        #    print(woe_var,"inf_rate kinds > 1")
        cate_id = d3.index[0]
        for cate_id in list(d3.index):
            d1.loc[d1["X"] == cate_id, "X"] = d3.loc[cate_id,'woe']
            d1_test.loc[d1_test["X"] == cate_id, "X"] = d3.loc[cate_id,'woe']
            
        train[woe_var] = d1['X']
        test[woe_var] = d1_test['X']
        
    return (train, test)

def cv_process(model_data, model_train, k):
    global cv_res
    global F1
    global AUC
    cv_res = []
    cv_index = np.array((list(range(k))*len(model_data))[:len(model_data)])
    F1 = []
    AUC = []
    for cv_i in set(cv_index):
        dataset_train = model_data[cv_index!=cv_i]
        dataset_test = model_data[cv_index==cv_i]
        
        # bins
        if ~bin_after_split: 
            data_train, data_test = lr_var_split(dataset_train, dataset_test)
            if raw_feature_iv_screen:
                iv = pd.DataFrame(data_iv(data_train,woe_vars = list(set(data_train.columns).difference(set(['y','user_id'])))),columns=['name','iv'])
                iv_low_name = list(iv.loc[iv['iv']<iv_threshold,'name'])
                data_train.drop(iv_low_name,axis=1,inplace=True)
                data_test.drop(iv_low_name,axis=1,inplace=True)
            else: 
                print("Not raw_feature_iv_screen")
            # woe tranform
            if use_woe_feature:
                data_train, data_test = data_woe(data_train, data_test, woe_vars = list(set(data_train.columns).difference(set(['y','user_id']))))
            else:
                print("Not use_woe_feature")
        
        # balanced data id
        split_id = balance_data(dataset_train[['user_id','y']])
        
        # build balanced model
        preds = []
        data_id = split_id[0]
        for data_id in split_id:
            if bin_after_split:
                data_train_pre = data_id.merge(dataset_train.drop(['y'],axis=1),how='inner',on='user_id',validate='one_to_one')
                data_test_pre = dataset_test
                data_train, data_test = lr_var_split(data_train_pre, data_test_pre)
                if raw_feature_iv_screen:
                    iv = pd.DataFrame(data_iv(data_train,woe_vars = list(set(data_train.columns).difference(set(['y','user_id'])))),columns=['name','iv'])
                    iv_low_name = list(iv.loc[iv['iv']<iv_threshold,'name'])
                    data_train.drop(iv_low_name,axis=1,inplace=True)
                    data_test.drop(iv_low_name,axis=1,inplace=True)
                else: 
                    print("Not raw_feature_iv_screen")
                
                if use_woe_feature:
                    data_train, data_test = data_woe(data_train, data_test, woe_vars = list(set(data_train.columns).difference(set(['y','user_id']))))
                else:
                    print("Not use_woe_feature")
            
            # onehot
            if use_woe_feature:
                train_onehot, test_onehot = data_train, data_test
            else:
                print('Not use_woe_feature')
                train_onehot, test_onehot = lr_data_onehot(data_train, data_test)
            
            lgb_data = data_id.merge(dataset_train.drop(['y'],axis=1),how='inner',on='user_id',validate='one_to_one')
            lr_data = data_id.merge(train_onehot.drop(['y'],axis=1),how='inner',on='user_id',validate='one_to_one')
    
            lgb_test = dataset_test
            lr_test = test_onehot
            
            pred = model_train(lgb_data, lr_data, lgb_test, lr_test)
            if len(preds)==0:
                preds = pred
            else:
                preds = preds.merge(pred,how='inner',on='user_id')
                
            
            
        ensemble_prob = pd.DataFrame(np.column_stack((preds['user_id'],np.mean(preds.drop(['user_id'],axis=1),axis=1))),columns=['user_id','prob'])
        temp = dataset_test[['user_id','y']].merge(ensemble_prob,how='inner',on='user_id')
        AUC.append(skl.metrics.roc_auc_score(temp['y'],temp['prob']))
        print('auc',AUC[-1])
        temp['pred'] = pd.DataFrame(skl.preprocessing.binarize(temp['prob'].values.reshape(-1,1),pos_neg_rate))
        F1.append(skl.metrics.f1_score(temp['y'],temp['pred']))
        print('f1',F1[-1])
            
        
    return np.mean(F1),np.sqrt(np.var(F1)), np.mean(AUC),np.sqrt(np.var(AUC))




dataset = pd.read_csv('model_sample.csv')
categorical_feature = ['x_001','x_010','x_011','x_027','x_033'] + ['x_00'+str(i) for i in range(3,9+1)] + ['x_0'+str(i) for i in range(13,19+1)]
if use_handcraft_feature :
    dataset = lr_data_clean(dataset)
else:
    dataset = dataset.drop(['x_012'] + missing_90 + list(drop_var["name"]),axis=1)
model_data = dataset


cv_process(model_data, model_train, 10)


