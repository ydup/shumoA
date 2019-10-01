from math import atan
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn import preprocessing, decomposition

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
    return data

def divideField(data):
    '''
    divide the 2D x-y field into 6 place


    '''
    pass

def ConvHeight(data, kernel_r = 1):
    map2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                  bins=(np.arange(data.loc[:, 'X_relative'].min(), data.loc[:, 'X_relative'].max()+5, 5), 
                  np.arange(data.loc[:, 'Y_relative'].min(), data.loc[:, 'Y_relative'].max()+5, 5)), 
                  weights=data.loc[:, 'Effective_Height'].values
                                )
    idx2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                  bins=(np.arange(data.loc[:, 'X_relative'].min(), data.loc[:, 'X_relative'].max()+5, 5), 
                  np.arange(data.loc[:, 'Y_relative'].min(), data.loc[:, 'Y_relative'].max()+5, 5)), 
                  weights=data.index+1
                                )
    conv_Sum = np.zeros_like(idx2d)
    conv_Num = np.zeros_like(idx2d)

    kernelHeight = []
    kernelPos = []

    for idx in tqdm(data.index+1):
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
        
    data.loc[:, 'conv_sum_{0}'.format(kernel_r)] = kernelHeight
    data.loc[:, 'conv_pos_{0}'.format(kernel_r)] = kernelPos
    return data

    
    
    

    






