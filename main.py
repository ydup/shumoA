import matplotlib
matplotlib.use('Agg')
import datetime
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示
from zipfile import ZipFile
import pickle
# set the environment path to find Recommenders
import sklearn
import numpy as np 
import pandas as pd
import os
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
from math import atan
from sklearn import preprocessing
from sklearn import decomposition
from xgboost import plot_importance
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
def rec2polar(x, y):
    x, y = float(x), float(y)
    R = np.sqrt(x**2 + y**2)  # the length from (tmp_x, tmp_y) to (0, 0)
    if x == 0:
        # the point is at y axis
        theta = np.sign(y)*np.pi/2
    elif x > 0:
        # the point is at I and IV regions
        theta = atan(y/x)
    elif x < 0:
        # the point is at II and III regions
        theta = np.sign(y)*np.pi+atan(y/x)
        if y == 0:
            theta = -np.pi
    return R, theta

def ConvHeight(data, kernel_r = 1):
    map2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                  bins=(np.arange(data.loc[:, 'X_relative'].min()-10.0, data.loc[:, 'X_relative'].max()+10.0, 5.0), 
                  np.arange(data.loc[:, 'Y_relative'].min()-10.0, data.loc[:, 'Y_relative'].max()+10.0, 5.0)), 
                  weights=data.loc[:, 'Effective_Height'].values
                                )
    idx2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                  bins=(np.arange(data.loc[:, 'X_relative'].min()-10.0, data.loc[:, 'X_relative'].max()+10.0, 5.0), 
                  np.arange(data.loc[:, 'Y_relative'].min()-10.0, data.loc[:, 'Y_relative'].max()+10.0, 5.0)), 
                  weights=data.index+1
                                )
    conv_Sum = np.zeros_like(idx2d)
    conv_Num = np.zeros_like(idx2d)

    kernelHeight = []
    kernelPos = []
    print(np.max(idx2d))
    for idx in data.index+1:
        center = np.argwhere(idx2d==idx)
        # print(center, idx)
        param_grid = {'x': np.arange(center[0][0]-kernel_r, center[0][0]+kernel_r+1, 1), 'y': np.arange(center[0][1]-kernel_r, center[0][1]+kernel_r+1, 1)}
        param = list(ParameterGrid(param_grid))
        kernelSum = 0
        posNum = 0
        for p in param:
            try:
                relative_center = map2d[p['x'], p['y']] - map2d[center[0][0], center[0][1]]
            except:
                relative_center = 0  # out of bound
            if relative_center > 0:
                kernelSum += relative_center
                posNum += 1

        kernelHeight.append(kernelSum)
        kernelPer = posNum / ((kernel_r*2+1)**2-1)
        kernelPos.append(kernelPer)
        conv_Sum[center[0][0], center[0][1]] = kernelSum
        conv_Num[center[0][0], center[0][1]] = kernelPer
        
    data.loc[:, 'conv_sum_{0}'.format(kernel_r)] = kernelSum
    data.loc[:, 'conv_pos_{0}'.format(kernel_r)] = kernelPer
    return data

def add_feature(data):
    '''Add new features into data'''
    data.loc[:, 'Effective_Cell_Height'] = data.loc[:, 'Height'] + data.loc[:, 'Cell Altitude']
    data.loc[:, 'Effective_Height'] = data.loc[:, 'Altitude'] + data.loc[:, 'Building Height']
    data.loc[:, 'seperation_distance'] = np.sqrt((data.loc[:, 'X'] - data.loc[:, 'Cell X'] - 2.5)**2 + (data.loc[:, 'Y'] - data.loc[:, 'Cell Y'] - 2.5)**2)
    data.loc[:, 'dh'] = np.log10(1+data.loc[:, 'seperation_distance'])*np.log10(1+data.loc[:, 'Effective_Cell_Height'])
    data.loc[:, 'X_relative'] = data.loc[:, 'X'] - data.loc[:, 'Cell X'] - 2.5
    data.loc[:, 'Y_relative'] = data.loc[:, 'Y'] - data.loc[:, 'Cell Y'] - 2.5
    data.loc[:, 'deltaH'] = data.loc[:, 'Effective_Cell_Height'] - data.loc[:, 'Effective_Height'] - data.loc[:, 'seperation_distance']*np.tan(np.pi/180.0*(data.loc[:, 'Electrical Downtilt']+data.loc[:, 'Mechanical Downtilt']))
    deltaTheta = []
    for x, y, t in zip(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, data.loc[:, 'Azimuth'].values):
        _, theta = rec2polar(float(x), float(y))
        # print(x, y, t, theta)
        deltaTheta.append(t - (90 - theta*180.0/np.pi))
    data.loc[:, 'deltaTheta'] = deltaTheta
    tempdata = preprocessing.scale(data.drop('RSRP', axis=1).values, axis=0)
    pca = decomposition.PCA(n_components=3)
    pcafeature = pca.fit_transform(tempdata)
    for idx, name in enumerate(['pca{0}'.format(i) for i in range(3)]):
        # data.loc[:, ['pca{0}'.format(i) for i in range(3)]]=pcafeature
        data.loc[:, name] = pcafeature[:, idx]
    clutter_class = np.unique(data.loc[:, 'Clutter Index'].values)
    clutter = []
    for cl in data.loc[:, 'Clutter Index'].values:
        clutter.append(np.array(cl == np.array(clutter_class), dtype=np.int32))
    clutter = np.array(clutter)
    for idx, name in enumerate(['one_hot{0}'.format(i) for i in range(len(clutter_class))]):    
        # data.loc[:, ['pca{0}'.format(i) for i in range(3)]]=pcafeature
        data.loc[:, name] = clutter[:, idx]
    # data = data.drop(['X', 'Y', 'Cell X', 'Cell Y', 'Clutter Index'], axis=1) 
    data = ConvHeight(data, 1)
    data = ConvHeight(data, 3)
    data = ConvHeight(data, 5)
    return data

def get_data(data_path):
    list_names=os.listdir(data_path)
    data=pd.read_csv(os.path.join(data_path,list_names[0]))
    print(list_names[0])
    data=add_feature(data)
    for i in tqdm(range(1,500)):
        one_data=pd.read_csv(os.path.join(data_path,list_names[i]))
        one_data=add_feature(one_data)
        data=pd.concat((data,one_data), axis=0)
        #if i%100==0:
        #    print(i)
    return data

data=get_data('./train_set/')

y=data['RSRP']
#y=np.log1p(-y)
X=data.drop('RSRP',axis=1)
# X=np.log1p(X)


#分割训练集和验证集
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y=train_test_split(X,y,test_size=0.1,random_state=1)

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import explained_variance_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from math import sqrt
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge


#lightgbm模型
import time
import lightgbm as lgb
'''
lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=7,
                              learning_rate=0.05, n_estimators=200,
                              max_bin = 50, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
begin_time=time.time()
lgb_model.fit(X,y)
print('消耗时间 %f s' % (time.time()-begin_time))

#开始验证
X_predictions = lgb_model.predict(val_X)

#mae=mean_absolute_error(-np.expm1(X_predictions),-np.expm1(val_y.values))
Mae=mean_absolute_error(X_predictions,val_y.values)

#rmse=sqrt(mean_squared_error(-np.expm1(X_predictions),-np.expm1(val_y.values)))
Rmse=sqrt(mean_squared_error(X_predictions,val_y.values))
#输出验证mae,rmse
#print("mae : %.4g" % mae)
print("Mae : %.4g" % Mae)
#print("rmse : %.4g" % rmse)
print("Rmse : %.4g" % Rmse)
'''


# xgboost模型
xgb_model = XGBRegressor(booster='gbtree',
                    objective= 'reg:linear',
                    eval_metric='rmse',
                    gamma = 0.05,
                    min_child_weight=3,
                    max_depth= 11,
                    subsample= 0.8,
                    colsample_bytree= 0.8,
                    tree_method= 'exact',
                    learning_rate=0.1,
                    n_estimators=30,
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

# fit rf_model_on_full_data on all data from the training data
begin_time=time.time()
#score=cv_rmse(xgb_model,X,y)
xgb_model.fit(X,y)
print('消耗时间 %f s' % (time.time()-begin_time))

#开始验证
X_predictions = xgb_model.predict(val_X)
# plot_importance()
mae=mean_absolute_error(np.expm1(X_predictions),np.expm1(val_y.values))
Mae=mean_absolute_error(X_predictions,val_y.values)

rmse=sqrt(mean_squared_error(np.expm1(X_predictions),np.expm1(val_y.values)))
Rmse=sqrt(mean_squared_error(X_predictions,val_y.values))
#输出验证mae,rmse
print("mae : %.4g" % mae)
print("Mae : %.4g" % Mae)
print("rmse : %.4g" % rmse)
print("Rmse : %.4g" % Rmse)
