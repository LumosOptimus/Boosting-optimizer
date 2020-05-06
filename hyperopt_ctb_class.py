# -*- coding: utf-8 -*-

from catboost import CatBoostRegressor, Pool, cv
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import numpy as np


cat_opt_parameters = {
    'learning_rate':       hp.choice('learning_rate',       np.arange(0.0001, 0.09, 0.0001)),
    'max_depth':           hp.choice('max_depth',           np.arange(6, 16, 1, dtype = int)),
    'random_strength':     hp.choice('random_strength',     np.arange(0.5, 0.9, 0.1)),
    'bagging_temperature': hp.choice('bagging_temperature', np.arange(0.5, 0.9, 0.1)),
    'eval_metric':         'MAPE',
    'l2_leaf_reg':         3
    }
cat_fit_parameters = {
    'early_stopping_rounds': 20,
    'logging_level':         'Silent'
    }
cat_params = {
    'opt_params':            cat_opt_parameters,
    'fit_params':            cat_fit_parameters,
    'loss_func':             lambda fact, pred: np.mean(np.abs((fact - pred) / fact) * 100)    # MAPE
    } 


class Hopt(object):

    def __init__(self, X, x_train, x_test, y, y_train, y_test, cat_features):
        self.X = X
        self.y = y
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        self.cat_features = cat_features
        self.pool = Pool(X, y, cat_features)
        self.train_pool = Pool(x_train, y_train, cat_features)
        self.test_pool = Pool(x_test, y_test, cat_features)
        
        
        
    def run(self, fn_name, space, max_evals, trials):
        fn = getattr(self, fn_name)
        res = fmin(
                fn = fn,
                space = space,
                algo = tpe.suggest,
                max_evals = max_evals,
                trials = trials
                )
        
        return res, trials
        
    
    
    def train_model(self, regressor, model_parameters):
        regressor.fit(
            self.train_pool, 
            eval_set = self.test_pool, 
            **model_parameters['fit_params']
            )
        pred = regressor.predict(self.x_test)
        loss = model_parameters['loss_func'](self.y_test, pred)
        return {
            'loss' : loss, 
            'status': STATUS_OK
            }
        
    
       
    def cross_val_model(self, regressor, model_parameters, nfolds):
       cv_data = cv(
           self.pool,
           model_parameters['opt_params'],
           fold_count = nfolds,
           logging_level = 'Silent'
           ) 
       best_acc = np.min(cv_data['test-RMSE-mean'])
       return {
           'loss': best_acc,
           'status': STATUS_OK
           }
        
   
    
    def ctb_regressor(self, model_parameters):
        regressor = CatBoostRegressor(**model_parameters['opt_params'])
        return self.train_model(regressor, model_parameters) 
    
    
    
    def ctb_regressor_gpu(self, model_parameters):
        regressor = CatBoostRegressor(**model_parameters['opt_params'],
                                      task_type = 'GPU')
        return self.train_model(regressor, model_parameters)
    
    
    
    def ctb_regressor_cv(self, model_parameters):
        regressor = CatBoostRegressor(**model_parameters['opt_params'], 
                                      loss_function = 'RMSE')
        return self.cross_val_model(regressor, model_parameters, 5) 
    
    
    
    def ctb_regressor_cv_gpu(self, model_parameters):
        # CV on GPU doesn't work
        regressor = CatBoostRegressor(**model_parameters['opt_params'], 
                                      loss_function = 'RMSE',
                                      task_type = 'GPU')
        return self.cross_val_model(regressor, model_parameters, 5) 
    
    
###################################################### 
#####################    usage    ####################
###################################################### 
       
obj = Hopt(X, x_train, x_test, y, y_train, y_test, cat_features)
cat_opt = obj.run(fn_name = 'ctb_regressor_gpu', space = cat_params, trials = Trials(), max_evals = 100)      
