
import datetime
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn import preprocessing
from sklearn import decomposition
from xgboost import plot_importance
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import explained_variance_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from math import sqrt
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
import time

def Invilidation(predictions, targets):
    val_class, pre_class = np.zeros_like(targets), np.zeros_like(targets)
    val_class[targets<-103] = 1.
    pre_class[predictions<-103] = 1.
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(val_class)):
        if val_class[i]==1. and pre_class[i]==1.:
            TP=TP+1
        if val_class[i]==0. and pre_class[i]==1.:
            FP=FP+1
        if val_class[i]==1. and pre_class[i]==0.:
            FN=FN+1
        if val_class[i]==0. and pre_class[i]==0.:
            TN=TN+1
    Recall=TP/(TP+FN)
    Precision =TP/(TP+FP)
    PCRR=2*Precision*Recall/(Precision+Recall)
    return Recall, Precision, PCRR

data = []
for i in tqdm(range(28)):
    data.append(pd.read_csv('feature/new_feature_receive_'+str(int(i))+'.csv'))

data = pd.concat(data, axis=0)
data.index = np.arange(data.shape[0])

data = data.loc[:, ['Azimuth_2', 'Azimuth_3', 'Height_0', 'Azimuth_1', 'Azimuth_4', 'Altitude', 
    'neighbor_num_3000', 'Azimuth_0', 'neighbor_num_4000', 'neighbor_num_2000', 'neighbor_num_5000', 
    'nearest_distance_3', 'nearest_distance_4', 'nerest_Q', 'nearest_distance_2', 'nearest_distance_1', 
    'Y', 'nearest_distance_0', 'X', 'nearest_theta']]

# validate = pd.read_csv('test.csv')
percentage = 0.95
idx = list(range(data.shape[0]))
import random
random.seed(10)
random.shuffle(idx)
train = data.loc[idx[0: int(percentage*data.shape[0])], :]
validate = data.loc[idx[int(percentage*data.shape[0]): ], :]

trainX, trainY = train.drop('RSRP',axis=1), train['RSRP']
# trainX = np.log1p(trainX)

valX, valY = validate.drop('RSRP',axis=1), validate['RSRP']
print('read finished')

import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(objective='regression',
                              learning_rate=0.08, n_estimators=4000, 
                              max_bin = 200
                              )
begin_time=time.time()
#score=cv_rmse(xgb_model,X,y)
lgb_model.fit(trainX, trainY, eval_set=(valX, valY), early_stopping_rounds=2000)
print('Fitting: %f s' % (time.time()-begin_time))

valY_predict = lgb_model.predict(valX)

Rmse = sqrt(mean_squared_error(valY_predict, valY.values))

Recall, Precision, PCRR = Invilidation(valY_predict, valY.values)

# protocal = 2

print("Recall : %.4g" % Recall)
print("Precision : %.4g" % Precision)
print("PCRR : %.4g" % PCRR)
print("Rmse : %.4g" % Rmse)

import pickle
with open('lgb.pkl', 'wb') as fw:
    pickle.dump(lgb_model, fw, protocal=2)






