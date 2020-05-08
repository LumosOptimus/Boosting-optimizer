# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:38:04 2020
@author: DMatveev
"""

import pandas as pd
import numpy as np
import logging

from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split

from hyperopt import Trials

#from preprocessing import preproc_with_extra_steps
import hyperopt_class

logging.basicConfig(filename = 'ctb_model_train.log')
split_line = '----------------------------------------------------------'
TRAIN_SIZE = 0.8


def split_and_cut_quantiles(dataset, features, train_size = TRAIN_SIZE, target_name = '', 
                            floor_quantile = 0.05, ceil_quantile = 0.95):
        '''Cut outliers by quantiles, split dataset to train and validation parts'''
        
        total_df = dataset[features].sample(frac = 1, random_state = 0)
        validation_df = total_df.iloc[int(np.ceil(len(total_df)*0.95)):, :]
        train_df = total_df.iloc[:int(np.ceil(len(total_df)*0.95)), :]
        
        train_df = train_df[(train_df[target_name] > train_df[target_name].quantile(floor_quantile)) &\
                            (train_df[target_name] < train_df[target_name].quantile(ceil_quantile))]
        
        
        X = train_df.drop(target_name, axis = 1)
        y = train_df[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state = 0)
        
        X_val = validation_df.drop(target_name, axis = 1)
        y_val = validation_df[target_name]
        return X, X_train, X_test, X_val, y, y_train, y_test, y_val

            



class ModelCTB(object):

    
    def __init__(self, X, X_train, X_test, X_val, y, y_train, y_test, y_val):
        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.y = y
        self.y_train = y_train
        self.y_test = y_test
        
        self.X_val = X_val
        self.y_val = y_val
        
        self.cat_features = list(X.columns.get_indexer(X.select_dtypes('object').columns))
        self.pool = Pool(X, y, self.cat_features)
        self.train_pool = Pool(X_train, y_train, self.cat_features)
        self.test_pool = Pool(X_test, y_test, self.cat_features)
        

                
    def mape(self, pred, fact):
        '''eval metric'''
        return np.mean((np.abs((fact - pred)/fact) * 100))
        


    def model_train(self, iterations = 1000, learning_rate = 0.001, random_strength = 0.8, bagging_temperature = 0.8, max_depth = 6):
        '''CatBoost Regression model'''
        date = pd.datetime.today().date()
        
        print('{}\nStart model learning proccess'.format(split_line))
        model = CatBoostRegressor(
            learning_rate = learning_rate,  
            random_strength = random_strength,
            bagging_temperature = bagging_temperature, 
            max_depth = max_depth,
            eval_metric = 'MAPE',
            loss_function = 'RMSE',  
            use_best_model = True,               
            iterations = iterations,
            l2_leaf_reg = 3, 
            early_stopping_rounds = 20,
            verbose = 200
            )
        
        model.fit(
            self.train_pool,
            eval_set = self.test_pool,
            plot = True
            )
        
        pred_train = np.round(self.mape(model.predict(self.X_train), self.y_train), 4)
        pred_test = np.round(self.mape(model.predict(self.X_test), self.y_test), 4)
        
        print('{}\nTrain MAPE: {}\nTest MAPE: {}'.format(split_line, pred_train, pred_test))
        
        model.save_model('catboost_models\catboost_model_{}_acc{}'.format(str(date), pred_test), format = 'cbm')
            
        return model    
            
      
        
    def feature_importance_review(self, model):
        '''Return feature importance score'''
        print(split_line)
        X = self.X
        feature_importances = model.get_feature_importance(self.train_pool)
        feature_score = sorted(zip(feature_importances, X.columns), reverse = True)
        
        for score, name in feature_score: 
            print('{}: {}'.format(name, score))
    
        return feature_score
    
    
    
    def model_cv(self, model, folds):
        '''Run model cross-validation'''
        cv_params = model.get_params()
        
        print('{}\nStart model crossvalidation proccess'.format(split_line))
        cv_data = cv(
            self.pool,
            cv_params,
            fold_count = folds,
            #iterations = 800,
            verbose = 200,
            early_stopping_rounds = 20
            )
        
        print('Best validation accuracy score: {:.4f}±{:.4f} on step {}'\
              .format(
                  np.min(cv_data['test-RMSE-mean']),
                  cv_data['test-RMSE-std'][np.argmax(cv_data['test-RMSE-mean'])],
                  np.argmax(cv_data['test-RMSE-mean'])
                  )
              )
    
    
    
    def model_validation(self, media = True, model_to_validate_name = ''):
        '''Validate model on previously defined validation set'''
        model_to_use = CatBoostRegressor().load_model(model_to_validate_name, 'cbm')
        X, y = self.X, self.y
        print('{}\nStart model validation proccess'.format(split_line))
        
        y_pred = model_to_use.predict(X)
        pred_score = np.round(self.mape(y_pred, y), 4)
        print('Validation MAPE: {}'.format(pred_score))
        
        if media:
            y_val_post, X_val_post = y[X[X['Format_type'] == 'Post'].index], X[X['Format_type'] == 'Post']
            y_val_video, X_val_video = y[X[X['Format_type'] == 'Video'].index], X[X['Format_type'] == 'Video']
            y_val_notvideo, X_val_notvideo = y[X[X['Format_type'] == 'notVideo'].index], X[X['Format_type'] == 'notVideo']
            
            y_pred_post = model_to_use.predict(X_val_post)
            y_pred_video = model_to_use.predict(X_val_video)
            y_pred_notvideo = model_to_use.predict(X_val_notvideo)
            
            pred_post_score = np.round(self.mape(y_pred_post, y_val_post), 4)
            pred_video_score = np.round(self.mape(y_pred_video, y_val_video), 4)
            pred_notvideo_score = np.round(self.mape(y_pred_notvideo, y_val_notvideo), 4)
        
            print('Validation MAPE: {}\nValidation Post MAPE: {}\nValidation Video MAPE: {}\nValidation notVIdeo MAPE: {}'\
                  .format(pred_score, pred_post_score, pred_video_score, pred_notvideo_score))
