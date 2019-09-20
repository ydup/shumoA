'''
Process the global information


1. The actual transmitter
2. The actual RPRS

'''

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Read all the data 
def gen_data(data_path, dst_path='train.csv'):
    list_names=os.listdir(data_path)
    data = pd.read_csv(os.path.join(data_path, list_names[0]))
    data.to_csv(dst_path, index=False)
    list_names = list_names[0: 6]
    for i in tqdm(range(1,  len(list_names)), desc=dst_path):
        data=pd.read_csv(os.path.join(data_path,list_names[i]))
        data.to_csv(dst_path, index=False, header=False, mode='a+')

gen_data('./train_set/', 'train.csv')
gen_data('./test_set/', 'test.csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Process the transmitter information

transmitter_train = train.loc[:, ['Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
            'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
            'Cell Clutter Index', 'Cell X', 'Cell Y']]
transmitter_test = test.loc[:, ['Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
            'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
            'Cell Clutter Index', 'Cell X', 'Cell Y']]
transmitter = pd.concat([transmitter_train, transmitter_test])

# Add the location string
transmitter.loc[:, 'Location'] = ['{0}-{1}'.format(*[int(x), int(y)]) 
                              for x, y in zip(transmitter.loc[:, 'Cell X'].values, transmitter.loc[:, 'Cell Y'].values)]

# Group the transmitter by location
gp_transmitter = transmitter.groupby('Location')

Flag = 0

for locate, info in tqdm(gp_transmitter, desc='group transmitter'):
    tmp = info.mean()
    tmp_pd = pd.DataFrame(tmp).T
    if Flag == 0:
        tmp_pd.to_csv('transmitter.csv', index=False)
        Flag = 1
    else:
        tmp_pd.to_csv('transmitter.csv', index=False, header=False, mode='a+')

receiver = train.drop(['Cell Index', 'Azimuth', 'Height', 'Electrical Downtilt', 'Mechanical Downtilt',
            'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height',
            'Cell Clutter Index', 'Cell X', 'Cell Y'], axis=1)

# Add the location string
receiver.loc[:, 'Location'] = ['{0}-{1}'.format(*[int(x), int(y)]) 
                              for x, y in zip(receiver.loc[:, 'X'].values, receiver.loc[:, 'Y'].values)]

# Group the transmitter by location
gp_receiver = receiver.groupby('Location')

Flag = 0

for locate, info in tqdm(gp_receiver, desc='group receiver'):
    tmp = info.mean()
    tmp_pd = pd.DataFrame(tmp).T
    if Flag == 0:
        tmp_pd.to_csv('receiver.csv', index=False)
        Flag = 1
    else:
        tmp_pd.to_csv('receiver.csv', index=False, header=False, mode='a+')







