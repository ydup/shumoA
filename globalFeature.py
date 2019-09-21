'''
Add global feature
'''
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.spatial import distance_matrix


transmitter = pd.read_csv('../../Downloads/transmitter.csv')
receiver =pd.read_csv('../../Downloads/receiver.csv')

# Add basic information

# Transmitter
# 		['Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
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
	return data

receiver = log_receiver(receiver)

# Calculate the distance matrix
# Shape: (receiver, transmitter)
print('Calculate distance matrix')
distance = distance_matrix(receiver.loc[:, ['X', 'Y']].values, transmitter.loc[:, ['Cell X', 'Cell Y']].values)
np.savetxt('distanceMatrix.txt', distance, delimiter=',')
print('Distance Matrix Saved')

def get_nearestFeature(distanceMatrix, transmitter, top_n_num=4):
	'''(Receiver, transmitter)'''
	top_n_index = np.argsort(distanceMatrix, axis=1)[:, 0: top_n_num]  # find the nearest samples # nearest index
	top_n_value = np.sort(distanceMatrix, axis=1)[:, 0: top_n_num]  # nearest distance
	topFeature = []
	for i in tqdm(range(distanceMatrix.shape[0]), desc='Add nearest feature'):
	    topTransmitter = transmitter.loc[top_n_index[i], ]
	    topTransmitter = topTransmitter.drop(['Cell X', 'Cell Y'], axis=1)
	    flatten = np.reshape(topTransmitter.T.values, [1, -1])
	    topFeature.append(flatten)
	topFeature = np.concatenate([np.concatenate(topFeature, axis=0), top_n_value], axis=1)
	columns = np.reshape([['{0}_{1}'.format(*[origin, top_n]) for top_n in range(top_n_num)] for origin in list(topTransmitter.columns)],
           [1, -1])
	columns = np.append(columns, ['nearest_distance_{0}'.format(top_n) for top_n in range(top_n_num)])
	return pd.DataFrame(topFeature, columns=columns)

# Add nearest information of transmitter
nearestFeature = get_nearestFeature(distance, transmitter, top_n_num=5)
receiver = pd.concat([receiver, nearestFeature], axis=1)

def get_neighborNum(distanceMatrix, thres=10000):
	thresMask = distanceMatrix < thresMask
	return np.sum(thresMask, axis=1)

for thres in [500, 1000, 2000, 3000, 4000, 5000]:
	receiver.loc[:, 'nearest_num_{0}'.format(thres)] = get_neighborNum(distanceMatrix, thres)

# One-hot encoding

def add_clutter_type(data):
    nameList = ['water_index', 'broaden_index', 'plant_index', 'bulding_index']
    for name, start in zip(nameList, [0, 3, 6, 9]):
        tmp = np.zeros_like(data.loc[:, 'Clutter Index'])
        tmp[(start<data.loc[:, 'Clutter Index'].values)&(data.loc[:, 'Clutter Index'].values<start+4)] = 1
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
	return pd.concat([receiver, nearestFeature], axis=1)

receiver = one_hot_encoding(receiver)

receiver.to_csv('new_feature_receive.csv', index=False)


