'''
Add feature
'''
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial import distance_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpisize = int(comm.Get_size())  # total num of the cpu cores, the n_splits of the k-Fold
mpirank = int(comm.Get_rank())  # rank of this core

# use 28 kernel

transmitter = pd.read_csv('./transmitter.csv')
receiver =pd.read_csv('./receiver.csv')

def chunk(data, batchsize):
    num = int(len(data)/batchsize)+1
    for i in range(num):
        yield data[i*batchsize: (i+1)*batchsize]

idxList = [idx for idx in chunk(range(receiver.shape[0]), int(receiver.shape[0]/28))]
receiver = receiver.loc[idxList[mpirank], :]
receiver.index = np.arange(receiver.shape[0])
# Add basic information

# Transmitter
#       ['Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
#       'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
#       'Cell Clutter Index', 'Cell X', 'Cell Y']

def log_transmitter(data):
    # drop the Clutter Index
    data = data.drop(['Cell Clutter Index'], axis=1)
    # add effective Height
    data.loc[:, 'Effective_Height'] = data.loc[:, 'Cell Altitude'] + data.loc[:, 'Cell Building Height'] + data.loc[:, 'Height']
    # log transform for RS and FB
    data.loc[:, 'RS Power'] = np.log10(data.loc[:, 'RS Power'])
    data.loc[:, 'Frequency Band'] = np.log10(data.loc[:, 'Frequency Band'])
    # log transform for the distance / Height
    data.loc[:, 'Effective_Height'] = np.log10(1 + data.loc[:, 'Effective_Height'])
    data.loc[:, 'Height'] = np.log10(1 + data.loc[:, 'Height'])
    data.loc[:, 'Cell Altitude'] = np.log10(1 + data.loc[:, 'Cell Altitude'])
    data.loc[:, 'Cell Building Height'] = np.log10(1 + data.loc[:, 'Cell Building Height'])
    return data

transmitter = log_transmitter(transmitter)

# Receiver
# ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index', 'RSRP']

def log_receiver(data):
    # Select the feature, suit for test dataset
    data = data.loc[:, ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index', 'RSRP']]
    # Add Effective_Height
    data.loc[:, 'Effective_Height'] = data.loc[:, 'Altitude'] + data.loc[:,'Building Height']
    # Log transform for the Height / distance
    data.loc[:, 'Altitude'] = np.log10(1 + data.loc[:, 'Altitude'])
    data.loc[:, 'Building Height'] = np.log10(1 + data.loc[:, 'Building Height'])
    data.loc[:, 'Effective_Height'] = np.log10(1 + data.loc[:, 'Effective_Height'])
    return data

receiver = log_receiver(receiver)

# Calculate the distance matrix
# Shape: (receiver, transmitter)
def get_neighborNum(distanceMatrix, thres=10000):
    thresMask = distanceMatrix < thres
    return np.sum(thresMask)

def rec2polar_array(x, y):
    R = np.sqrt(x**2 + y**2)  # the length from (tmp_x, tmp_y) to (0, 0)
    # the point is at y axis
    theta = np.zeros_like(R)
    theta[x == 0] = np.sign(y[x == 0])*np.pi/2
    # the point is at I and IV regions
    theta[x > 0] = np.arctan(y[x > 0]/x[x > 0])
    # the point is at II and III regions
    theta[x < 0] = np.sign(y[x < 0])*np.pi+np.arctan(y[x < 0]/x[x < 0])
    return R, theta

def get_nearestFeature(receiver, transmitter, top_n_num=4):
    '''(Receiver, transmitter)'''
    topFeature = []
    thresList = [500, 1000, 2000, 3000, 4000, 5000]
    for i in tqdm(range(receiver.shape[0]), desc='Add nearest feature'):
        recX, recY = receiver.loc[i, ['X', 'Y']].values
        tranX, tranY = transmitter.loc[:, 'Cell X'].values, transmitter.loc[:, 'Cell Y'].values
        distanceMatrix = np.sqrt((recX - tranX)**2 + (recY - tranY)**2)
        top_n_index = np.argsort(distanceMatrix)[0: top_n_num]  # find the nearest samples # nearest index
        top_n_value = np.sort(distanceMatrix)[0: top_n_num]  # nearest distance
        topTransmitter = transmitter.loc[top_n_index, ]
        topTransmitter = topTransmitter.drop(['Cell X', 'Cell Y'], axis=1)
        flatten = np.reshape(topTransmitter.T.values, [1, -1])
        # Neibor num
        neighborNum = []
        for thres in thresList:
            neighborNum.append(get_neighborNum(distanceMatrix, thres))
        flatten = np.append(flatten, top_n_value)  # concat top_n_value list
        flatten = np.append(flatten, np.array(neighborNum))  # concat neighborNum list
        # R and theta
        topRSPower = topTransmitter.loc[top_n_index, 'RS Power'].values
        topdeltaX = tranX[top_n_index] - recX
        topdeltaY = tranY[top_n_index] - recY
        topQ, toptheta = rec2polar_array(topdeltaX/topRSPower, topdeltaY/topRSPower)
        flatten = np.append(flatten, [np.mean(topQ), np.mean(toptheta)])
        # append to top feature
        topFeature.append(flatten)
    topFeature = np.stack(topFeature, axis=0)
    columns = np.reshape([[str(origin)+'_'+str(int(top_n)) for top_n in range(top_n_num)] for origin in list(topTransmitter.columns)],
           [1, -1]) 
    columns = np.append(columns, ['nearest_distance_'+str(int(top_n)) for top_n in range(top_n_num)])
    columns = np.append(columns, ['neighbor_num_'+str(int(thres)) for thres in thresList])
    columns = np.append(columns, ['nerest_Q', 'nearest_theta'])
    return pd.DataFrame(topFeature, columns=columns)

# Add nearest information of transmitter
nearestFeature = get_nearestFeature(receiver, transmitter, top_n_num=5)
receiver = pd.concat([receiver, nearestFeature], axis=1)


# One-hot encoding

def add_clutter_type(data):
    nameList = ['water_index', 'broaden_index', 'plant_index', 'bulding_index']
    for name, start in zip(nameList, [0, 3, 6, 9]):
        tmp = np.zeros_like(data.loc[:, 'Clutter Index'])
        tmp[(start<=data.loc[:, 'Clutter Index'].values)&(data.loc[:, 'Clutter Index'].values<start+3)] = 1
        data.loc[:, name] = tmp
    return data

def get_clutter_one_hot(data):
    clutter_class = np.arange(1, 21)
    clutter = []
    for cl in data.loc[:, 'Clutter Index'].values:
        clutter.append(np.array(cl == np.array(clutter_class), dtype=np.float32))
    clutter = pd.DataFrame(np.array(clutter), columns=['clutter_one_hot'+str(int(i)) for i in clutter_class])
    return clutter
    
def one_hot_encoding(data):
    '''for clutter index'''
    data = add_clutter_type(data)
    clutter = get_clutter_one_hot(data)
    return pd.concat([data, clutter], axis=1)

receiver = add_clutter_type(receiver)

receiver.to_csv('new_feature/new_feature_receive_{0}.csv'.format(int(mpirank)), index=False)


