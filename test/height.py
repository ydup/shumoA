'''
Prepare effective height for receiver
'''
import pandas as pd
from tqdm import tqdm
import numpy as np

data = []
for i in tqdm(range(28)):
    data.append(pd.read_csv('new_feature/new_feature_receive_'+str(int(i))+'.csv'))

data = pd.concat(data, axis=0)
data.index = np.arange(data.shape[0])

height = data.loc[:, ['X', 'Y', 'Effective_Height']]

height.loc[:, 'Effective_Height'] = np.power(height.loc[:, 'Effective_Height'].values, 10) - 1

height.index = [str(int(x))+'_'+str(int(y)) for x, y in height.loc[:, ['X', 'Y']].values]

height = height.drop(['X', 'Y'], axis=1)

height.to_csv('height.csv')
