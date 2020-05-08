import pandas as pd
import numpy as np
import logging

from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.ensemble import IsolationForest
#from sklearn.svm import OneClassSVM
from hyperopt import Trials

#from preprocessing import preproc_with_extra_steps
import hyperopt_class

logging.basicConfig(filename = 'ctb_model_train.log')


split_line = '----------------------------------------------------------'

class ModelCTB(object):
    
    train_size = 0.8
    
    def __init__(self, dataset, features):
        dataset = dataset
        features = features
        self.total_df = dataset[features].sample(frac = 1, random_state = 0)
        self.validation_df = self.total_df.iloc[int(np.ceil(len(self.total_df)*0.95)):, :]
        self.train_df = self.total_df.iloc[:int(np.ceil(len(self.total_df)*0.95)), :]
        self.categorical_features = list(self.total_df.columns.get_indexer(self.total_df.select_dtypes('object').columns))[:]

        
        
    def mape(self, pred, fact):
        '''eval metric'''
        return np.mean((np.abs((fact - pred)/fact) * 100))
        


    def split_info(self):
        '''Return dataset fomat_type statistics'''
        
        train_post, validation_post = self.train_df.query('Format_type == "Post"'), self.validation_df.query('Format_type == "Post"')
        train_video, validation_video = self.train_df.query('Format_type == "Video"'), self.validation_df.query('Format_type == "Video"')
        train_notvideo, validation_notvideo = self.train_df.query('Format_type == "notVideo"'), self.validation_df.query('Format_type == "notVideo"')
        
        def calc_stats(numerator, denominator):
            return np.round(len(numerator)/len(denominator)*100, 0)
        
        train_observations = len(self.train_df)
        train_percentage = calc_stats(self.train_df, self.total_df)
        train_post_percentage = calc_stats(train_post, self.train_df)
        train_video_percentage = calc_stats(train_video, self.train_df)
        train_notvideo_percentage = calc_stats(train_notvideo, self.train_df)
        
        validation_observations = len(self.validation_df)
        validation_percentage = calc_stats(self.validation_df, self.total_df)
        validation_post_percentage = calc_stats(validation_post, self.validation_df)
        validation_video_percentage = calc_stats(validation_video, self.validation_df)
        validation_notvideo_percentage = calc_stats(validation_notvideo, self.validation_df)
        
        print('Observations for train: {} - {}%, where:\n-post: {}% \n-video: {}% \n-notvideo: {}%\
              \nObservations for validation: {} - {}%, where:\n-post: {}% \n-video: {}% \n-notvideo: {}%'\
              .format(train_observations, train_percentage, train_post_percentage, train_video_percentage, train_notvideo_percentage, \
                      validation_observations, validation_percentage, validation_post_percentage, validation_video_percentage, validation_notvideo_percentage,))
    


    def split_and_cut_quantiles(self, target_name = '', floor_quantile = 0.05, ceil_quantile = 0.95, train = True, validation= False):
        '''Cut outliers by quantiles, split dataset to train and validation parts'''
        train_df = self.train_df
        validation_df = self.validation_df
        
        train_df = train_df[(train_df[target_name] > train_df[target_name].quantile(floor_quantile)) &\
                            (train_df[target_name] < train_df[target_name].quantile(ceil_quantile))]
        
        if train:
            X = train_df.drop(target_name, axis = 1)
            y = train_df[target_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = self.train_size, random_state = 0)
            return X, X_train, X_test, y, y_train, y_test
        
        elif validation:
            X = validation_df.drop(target_name, axis = 1)
            y = validation_df[target_name]
            return X, y



    def model_train(self, X, y, iterations = 1000, learning_rate = 0.001, random_strength = 0.8, bagging_temperature = 0.8, max_depth = 6):
        '''CatBoost Regression model'''
        date = pd.datetime.today().date()
        
        train_pool = Pool(X_train, y_train, cat_features = self.categorical_features)
        validate_pool = Pool(X_test, y_test, cat_features = self.categorical_features)
        
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
            train_pool,
            eval_set = validate_pool,
            plot = True
            )
        
        pred_train = np.round(self.mape(model.predict(X_train), y_train), 4)
        pred_test = np.round(self.mape(model.predict(X_test), y_test), 4)
        
        print('{}\nTrain MAPE: {}\nTest MAPE: {}'.format(split_line, pred_train, pred_test))
        
        model.save_model('catboost_models\catboost_model_{}_acc{}'.format(str(date), pred_test), format = 'cbm')
            
        return model    
            
      
        
    def feature_importance_review(self, X_train, y_train, model):
        '''Return feature importance score'''
        print(split_line)
        train_pool = Pool(X_train, y_train, cat_features = self.categorical_features)
        
        feature_importances = model.get_feature_importance(train_pool)
        feature_score = sorted(zip(feature_importances, X.columns), reverse = True)
        
        for score, name in feature_score: 
            print('{}: {}'.format(name, score))
    
        return feature_score
    
    
    
    def model_cv(self, X, y, model, folds):
        '''Run model cross-validation'''
        cv_params = model.get_params()
        pool = Pool(X, y, cat_features = self.categorical_features)
        
        print('{}\nStart model crossvalidation proccess'.format(split_line))
        cv_data = cv(
            pool,
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
    
    
    
    def model_validation(self, X, y, media = False, model_to_validate_name = ''):
        '''Validate model on previously defined validation set'''
        model_to_use = CatBoostRegressor().load_model(model_to_validate_name, 'cbm')
        
        print('{}\nStart model validation proccess'.format(split_line))
        
        y_pred = model_to_use.predict(X)
        pred_score = np.round(self.mape(y_pred, y), 4)
        print('Validation MAPE: {}'.format(pred_score))
        
        if media:
            pass
 
