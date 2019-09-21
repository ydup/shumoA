'''
Build up a DNN network to predict RS power

And embed the equal RS power into Receiver grid.

'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
transmitter = pd.read_csv('transmitter.csv')
transmitter.loc[:, 'Equ power'] = transmitter.loc[:, 'Frequency Band']*transmitter.loc[:, 'RS Power']
transmitter = transmitter.drop(['Frequency Band', 'RS Power', 'Azimuth', 'Electrical Downtilt', 
	'Mechanical Downtilt'], axis=1)

receiver = pd.read_csv('receiver.csv')

dataRange = {
'X': (np.min((np.min(receiver.loc[:, 'X']), np.min(transmitter.loc[:, 'Cell X']))),
		np.max((np.max(receiver.loc[:, 'X']), np.max(transmitter.loc[:, 'Cell X'])))), 
'Y': (np.min((np.min(receiver.loc[:, 'Y']), np.min(transmitter.loc[:, 'Cell Y']))),
		np.max((np.max(receiver.loc[:, 'Y']), np.max(transmitter.loc[:, 'Cell Y'])))),
'Height': (np.min((0, np.min(transmitter.loc[:, 'Height']))), 
		np.max((0,np.max(transmitter.loc[:, 'Height'])))),
'BuildingHeight': (np.min((np.min(receiver.loc[:, 'Building Height']), np.min(transmitter.loc[:, 'Cell Building Height']))),
		np.max((np.max(receiver.loc[:, 'Building Height']), np.max(transmitter.loc[:, 'Cell Building Height'])))),
'Altitude': (np.min((np.min(receiver.loc[:, 'Altitude']), np.min(transmitter.loc[:, 'Cell Altitude']))),
		np.max((np.max(receiver.loc[:, 'Altitude']), np.max(transmitter.loc[:, 'Cell Altitude'])))),
'Clutter': (0, 20),
'EquPower': (np.min(transmitter.loc[:, 'Equ power']), np.max(transmitter.loc[:,'Equ power']))
}

RSRPrange = (np.min(receiver.loc[:, 'RSRP'].values), np.max(receiver.loc[:, 'RSRP'].values))
print(RSRPrange)

nameDict = {
	'X': {'transmitter': 'Cell X', 'receiver': 'X'},
	'Y': {'transmitter': 'Cell Y', 'receiver': 'Y'},
	'Height': {'transmitter': 'Height', 'receiver': 'Height'},
	'BuildingHeight': {'transmitter': 'Cell Building Height', 'receiver': 'Building Height'},
	'Altitude': {'transmitter': 'Cell Altitude', 'receiver': 'Altitude'},
	'Clutter': {'transmitter': 'Cell Clutter Index', 'receiver': 'Clutter Index'},
	'EquPower': {'transmitter': 'Equ power', 'receiver': 'Equ power'}
}

transmitter_transform = pd.DataFrame(np.zeros_like(transmitter), columns=list(dataRange.keys()))

def minmax(data, minmaxscale):
	min_, max_ = minmaxscale[0], minmaxscale[1]
	return (data - min_)/(max_ - min_)

for key in tqdm(list(dataRange.keys())):
	transmitter_transform.loc[:, key] = minmax(transmitter.loc[:, nameDict[key]['transmitter']].values, dataRange[key])

# transmitter_transform.to_csv('transmitter_transform.csv', index=False)

# transmitter_transform = pd.read_csv('transmitter_transform.csv')
print(transmitter_transform)
X = transmitter_transform.drop(['EquPower'], axis=1)
Y = transmitter_transform['EquPower']
X, Y = X.values, Y.values

model = keras.Sequential([
    keras.layers.Dense(X.shape[1]),
    keras.layers.Dense(128), 
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

optimizer = keras.optimizers.Adam()
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_squared_error'])
nb_epochs = 1000
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=50, min_lr=0.00001) 

hist = model.fit(X, Y, batch_size=64, nb_epoch=nb_epochs,
          verbose=1, callbacks = [reduce_lr])

#Print the testing results which has the lowest training loss.
log = pd.DataFrame(hist.history)
# print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])
log.to_csv('log1.csv')

receiver.loc[:, 'Height'] = np.zeros_like(receiver.loc[:, 'X'])
receiver.loc[:, 'Equ power'] = np.zeros_like(receiver.loc[:, 'X'])

receiverNameList = list(dataRange.keys())
receiverNameList.append('RSRP')

print(receiver.columns)
receiver_transform = pd.DataFrame(np.zeros_like(receiver), columns=receiverNameList)
receiver_transform.loc[:, 'RSRP'] = minmax(receiver.loc[:, 'RSRP'].values, RSRPrange)

print(receiver_transform.columns)

for key in tqdm(list(dataRange.keys())):
	receiver_transform.loc[:, key] = minmax(receiver.loc[:, nameDict[key]['receiver']].values, dataRange[key])

receiver_equPower = model.predict(receiver_transform.drop(['RSRP', 'EquPower'], axis=1).values, verbose=1)
receiver_transform.loc[:, 'EquPower'] = receiver_equPower


X = receiver_transform.drop(['RSRP'], axis=1)
Y = receiver_transform['RSRP']
X, Y = X.values, Y.values

model = keras.Sequential([
    keras.layers.Dense(X.shape[1]),
    keras.layers.Dense(128), 
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

optimizer = keras.optimizers.Adam()
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_squared_error'])
nb_epochs = 1000
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=50, min_lr=0.00001) 

hist = model.fit(X, Y, batch_size=512, nb_epoch=nb_epochs,
          verbose=1, callbacks = [reduce_lr])

#Print the testing results which has the lowest training loss.
log = pd.DataFrame(hist.history)
# print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])
log.to_csv('log2.csv')


# Preprocess the transmitter data according to the information of Receiver


# MLP to predict RS power


# Get the trained network


# Preprocess the receiver data with the same scalers


# MLP to predict RSRP


# Get the trained network


# Predict on the target point









