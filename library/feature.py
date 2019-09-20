import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def normImg(image):
    return (image-np.min(image))/(np.max(image)-np.min(image))

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

def add_feature(data):
    '''Add new features into data'''
    data.loc[:, 'Effective_Cell_Height'] = data.loc[:, 'Height'] + data.loc[:, 'Cell Altitude']
    data.loc[:, 'Effective_Height'] = data.loc[:, 'Altitude'] + data.loc[:, 'Building Height']
    data.loc[:, 'seperation_distance'] = np.sqrt((data.loc[:, 'X'] - data.loc[:, 'Cell X'] - 2.5)**2 + (data.loc[:, 'Y'] - data.loc[:, 'Cell Y'] + 2.5)**2)
    data.loc[:, 'dh'] = np.log10(1+data.loc[:, 'seperation_distance'])*np.log10(1+data.loc[:, 'Effective_Cell_Height'])
    data.loc[:, 'X_relative'] = data.loc[:, 'X'] - data.loc[:, 'Cell X'] - 2.5
    data.loc[:, 'Y_relative'] = data.loc[:, 'Y'] - data.loc[:, 'Cell Y'] + 2.5
    data.loc[:, 'deltaH'] = data.loc[:, 'Effective_Cell_Height'] - data.loc[:, 'Effective_Height'] - data.loc[:, 'seperation_distance']*np.tan(np.pi/180.0*(data.loc[:, 'Electrical Downtilt']+data.loc[:, 'Mechanical Downtilt']))
    data.loc[:, 'distance_3D'] = np.sqrt(data.loc[:, 'seperation_distance']**2 + data.loc[:, 'deltaH']**2)
    data.loc[:, 'vertical_theta'] = rec2polar_array(data.loc[:, 'deltaH'].values, data.loc[:, 'seperation_distance'].values)*180.0/np.pi
    data.loc[:, 'horizon_theta'] = rec2polar_array(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values)
    deltaTheta, theta_relative = [], []
    thetaXY = 90 - data.loc[:, 'horizon_theta']*180.0/np.pi
    thetaXY[thetaXY<0] += 360
    data.loc[:, 'horizon_theta'] = thetaXY
    deltaTh = thetaXY - data.loc[:, 'Azimuth']
    deltaTh[deltaTh<0] += 360
    data.loc[:, 'deltaTheta'] = deltaTh
    return data

class preprocess():
    def __init__(self):
        self.bins = (np.arange(-5002.5, 5002.5, 5), np.arange(-5002.5, 5002.5, 5))

    def get_image_const(self, data):
        image = []
        var2d = data.drop(['Cell X', 'Cell Y', 'Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
                    'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
                    'Cell Clutter Index', 'X', 'Y', 'X_relative', 'Y_relative'], axis=1)
        mask, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, bins=self.bins)
        idx2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, bins=self.bins, weights=data.index+1)

        for name in var2d.columns:
            map2d, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                      bins=self.bins, weights=var2d.loc[:, name].values)
            image.append(normImg(map2d))
        image = np.stack(image, axis=2)  # x, y, channel
        const = data.loc[:, ['Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
                'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
                'Cell Clutter Index']]
        const = np.unique(const, axis=0)
        return image, mask, idx2d, const

    def get_target(self, data):
        target, _, _ = np.histogram2d(data.loc[:, 'X_relative'].values, data.loc[:, 'Y_relative'].values, 
                                      bins=self.bins, weights=data.loc[:, 'RSRP'].values)
        return normImg(target)










