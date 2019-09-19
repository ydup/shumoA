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
from tqdm import tqdm
import numpy as np 
import pandas as pd
import os
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
from feature import *

def get_data(data_path):
    list_names=os.listdir(data_path)
    data=pd.read_csv(os.path.join(data_path,list_names[0]))
    for i in tqdm(range(1,500)):
        one_data=pd.read_csv(os.path.join(data_path,list_names[i]))
        one_data=add_feature(one_data)
        data=pd.concat((data,one_data), axis=0)
        if i%100==0:
            print(i)
    return data

data=get_data('./train_set/')

y=data['RSRP']
#y=np.log1p(-y)
X=data.drop('RSRP',axis=1)
X=np.log1p(X)


#分割训练集和验证集
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y=train_test_split(X, y, test_size=0.1,random_state=1)

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