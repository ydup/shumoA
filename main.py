import matplotlib
matplotlib.use('Agg')
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
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
import time

def Invilidation(predictions, val_class):
    pre_class=np.zeros(len(val_class))
    for k in range(len(predictions)):
        if predictions[k]<-103:
            pre_class[k]=1.
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(len(val_class)):
        if val_class[i]==1. and pre_class[i]==val_class[i]:
            TP=TP+1
        if val_class[i]==0. and pre_class[i]==1.:
            FP=FP+1
        if val_class[i]==1. and pre_class[i]==0.:
            FN=FN+1
        if val_class[i]==0. and pre_class[i]==val_class[i]:
            TN=TN+1
    Recall=TP/(TP+FN)
    Precision =TP/(TP+FP)
    PCRR=2*Precision*Recall/(Precision+Recall)
    return Recall,Precision,PCRR

def rec2polar_array(x, y):
    R = np.sqrt(x**2 + y**2)  # the length from (tmp_x, tmp_y) to (0, 0)
    # the point is at y axis
    theta = np.zeros_like(R)
    theta[x == 0] = np.sign(y[x == 0])*np.pi/2
    # the point is at I and IV regions
    theta[x > 0] = np.arctan(y[x > 0]/x[x > 0])
    # the point is at II and III regions
    theta[x < 0] = np.sign(y[x < 0])*np.pi+np.arctan(y[x < 0]/x[x < 0])
    return theta

def sliceXY(data):
    deltaTheta = data.loc[:, 'deltaTheta'].values
    plane = np.zeros_like(deltaTheta)
    plane[((deltaTheta <= 45)&(deltaTheta > 0))|((deltaTheta <= 360)&(deltaTheta > 315))] = 1
    plane[(deltaTheta <= 90)&(deltaTheta > 45)] = 2
    plane[(deltaTheta <= 135)&(deltaTheta > 90)] = 3
    plane[((deltaTheta <= 180)&(deltaTheta > 135))|((deltaTheta <= 225)&(deltaTheta > 180))] = 4
    plane[(deltaTheta <= 270)&(deltaTheta > 225)] = 5
    plane[(deltaTheta <= 315)&(deltaTheta > 270)] = 6
    data.loc[:, 'XYslice'] = plane
    plane[plane==6] = 2
    plane[plane==5] = 3
    data.loc[:, 'XYpower'] = plane
    return data

def ConvHeight(data, kernel_r = 1):
    bins = (np.arange(data.loc[:, 'X_relative'].min()-10.0, data.loc[:, 'X_relative'].max()+10.0, 5.0), 
            np.arange(data.loc[:, 'Y_relative'].min()-10.0, data.loc[:, 'Y_relative'].max()+10.0, 5.0))
    map2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                  bins=bins, weights=data.loc[:, 'Effective_Height'].values)
    idx2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                  bins=bins, weights=data.index+1)
    conv_Sum = np.zeros_like(idx2d)
    conv_Num = np.zeros_like(idx2d)

    kernelHeight = []
    kernelPos = []
    for idx in data.index+1:
        center = np.argwhere(idx2d==idx)
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

def add_clutter_type(data):
    nameList = ['water_index', 'broaden_index', 'plant_index', 'bulding_index']
    for name, start in zip(nameList, [0, 3, 6, 9]):
        tmp = np.zeros_like(data.loc[:, 'Clutter Index'])
        tmp[(start<data.loc[:, 'Clutter Index'].values)&(data.loc[:, 'Clutter Index'].values<start+4)] = 1
        data.loc[:, name] = tmp
    return data

def add_feature(data):
    '''Add new features into data'''
    data.loc[:, 'Effective_Cell_Height'] = data.loc[:, 'Height'].values + data.loc[:, 'Cell Altitude'].values
    data.loc[:, 'Effective_Height'] = data.loc[:, 'Altitude'].values + data.loc[:, 'Building Height'].values
    data.loc[:, 'seperation_distance'] = np.sqrt((data.loc[:, 'X'].values - data.loc[:, 'Cell X'].values - 2.5)**2 + (data.loc[:, 'Y'].values - data.loc[:, 'Cell Y'].values + 2.5)**2)
    data.loc[:, 'dh'] = np.log10(1+data.loc[:, 'seperation_distance'].values)*np.log10(1+data.loc[:, 'Effective_Cell_Height'].values)
    data.loc[:, 'X_relative'] = data.loc[:, 'X'].values - data.loc[:, 'Cell X'].values - 2.5
    data.loc[:, 'Y_relative'] = data.loc[:, 'Y'].values - data.loc[:, 'Cell Y'].values + 2.5
    data.loc[:, 'deltaH'] = data.loc[:, 'Effective_Cell_Height'].values - data.loc[:, 'Effective_Height'].values - data.loc[:, 'seperation_distance'].values*np.tan(np.pi/180.0*(data.loc[:, 'Electrical Downtilt'].values + data.loc[:, 'Mechanical Downtilt'].values))

    data.loc[:, 'distance_3D'] = np.sqrt(data.loc[:, 'seperation_distance'].values**2 + data.loc[:, 'deltaH'].values**2)
    data.loc[:, 'vertical_theta'] = rec2polar_array(data.loc[:, 'deltaH'].values, data.loc[:, 'seperation_distance'].values)*180.0/np.pi
    data.loc[:, 'horizon_theta'] = rec2polar_array(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values)
    
    deltaTheta, theta_relative = [], []
    thetaXY = 90 - data.loc[:, 'horizon_theta'].values*180.0/np.pi
    thetaXY[thetaXY<0] += 360
    data.loc[:, 'horizon_theta'] = thetaXY
    deltaTh = thetaXY - data.loc[:, 'Azimuth'].values
    deltaTh[deltaTh<0] += 360
    data.loc[:, 'deltaTheta'] = deltaTh
    try:
        tempdata = preprocessing.scale(data.drop('RSRP', axis=1).values, axis=0)
    except: 
        tempdata = preprocessing.scale(data.values, axis=0)
        pass
    pca = decomposition.PCA(n_components=3)
    pcafeature = pca.fit_transform(tempdata)
    print(pcafeature.shape)
    for idx in np.arange(3):
        # data.loc[:, ['pca{0}'.format(i) for i in range(3)]]=pcafeature
        # ['pca{0}'.format(i) for i in range(3)]
        data.loc[:, 'pca_{0}'.format(idx)] = pcafeature[:, idx]
    
    clutter_class = np.arange(1, 21)
    clutter = []
    for cl in data.loc[:, 'Clutter Index'].values:
        clutter.append(np.array(cl == np.array(clutter_class), dtype=np.float32))
    clutter = np.array(clutter)
    for idx, name in enumerate(['one_hot'+str(int(i)) for i in range(len(clutter_class))]):    
        # data.loc[:, ['pca{0}'.format(i) for i in range(3)]]=pcafeature
        data.loc[:, name] = clutter[:, idx]
    
    # data = data.drop(['X', 'Y', 'Cell X', 'Cell Y', 'Clutter Index'], axis=1) 
    #data = ConvHeight(data, 1)
    #data = ConvHeight(data, 3)
    #data = ConvHeight(data, 5)
    data = add_clutter_type(data)
    
    return data


def gen_data(data_path, test_size=5):
    list_names=os.listdir(data_path)
    data=pd.read_csv(os.path.join(data_path, list_names[0]))
    data = add_feature(data)
    data.to_csv('train.csv', index=False)
    list_names = list_names[0: 6]
    for i in tqdm(range(1,  len(list_names))):
        data=pd.read_csv(os.path.join(data_path,list_names[i]))
        data=add_feature(data)
        if i < (len(list_names) - test_size):
            data.to_csv('train.csv', index=False, header=False, mode='a+')
        elif i == (len(list_names) - test_size):
            data.to_csv('test.csv', index=False)
        else:
            data.to_csv('test.csv', index=False, header=False, mode='a+')

gen_data('../../Downloads/train_set/')
train = pd.read_csv('train.csv')
validate = pd.read_csv('test.csv')

test = {'Name': [], 'Data': []}
for testfile in os.listdir('../../Downloads/test_set/'):
    test['Data'].append(np.log1p(add_feature(pd.read_csv(os.path.join('../../Downloads/test_set/',testfile)))))
    test['Name'].append(testfile)

trainX, trainY = train.drop('RSRP',axis=1), train['RSRP']
trainX = np.log1p(trainX)

valX, valY = validate.drop('RSRP',axis=1), validate['RSRP']
valX = np.log1p(valX)

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
                    n_estimators=300,
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)

begin_time=time.time()
#score=cv_rmse(xgb_model,X,y)
xgb_model.fit(trainX, trainY)
print('Fitting: %f s' % (time.time()-begin_time))

valY_predict = xgb_model.predict(valX)
# plot_importance()

Rmse = sqrt(mean_squared_error(valY_predict,valY.values))
val_class = np.zeros_like(valY)

val_class[valY.values<-103] = 1 
Recall, Precision, PCRR = Invilidation(valY_predict, val_class)

print("Recall : %.4g" % Recall)
print("Precision : %.4g" % Precision)
print("PCRR : %.4g" % PCRR)
print("Rmse : %.4g" % Rmse)
 
for name, data in zip(test['Name'], test['Data']):
    predict_test = xgb_model.predict(data)
    dict_output = {'RSRP': None}
    dict_output['RSRP'] = np.reshape(predict_test, [-1, 1]).tolist()
    print(dict_output)
    with open('{0}_result.txt'.format(name), "w") as f:
        f.write(str(dict_output))

