import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
import cPickle
import os


def log_receiver(data):
    # Select the feature, suit for test dataset
    data = data.loc[:, ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']]
    # Add Effective_Height
    data.loc[:, 'Effective_Height'] = data.loc[:, 'Altitude'] + data.loc[:,'Building Height']
    # Log transform for the Height / distance
    data.loc[:, 'Altitude'] = np.log10(1 + data.loc[:, 'Altitude'])
    data.loc[:, 'Building Height'] = np.log10(1 + data.loc[:, 'Building Height'])
    data.loc[:, 'Effective_Height'] = np.log10(1 + data.loc[:, 'Effective_Height'])
    return data

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
    for i in range(receiver.shape[0]):
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

def add_clutter_type(data):
    nameList = ['water_index', 'broaden_index', 'plant_index', 'bulding_index']
    for name, start in zip(nameList, [0, 3, 6, 9]):
        tmp = np.zeros_like(data.loc[:, 'Clutter Index'])
        tmp[(start<data.loc[:, 'Clutter Index'].values)&(data.loc[:, 'Clutter Index'].values<start+4)] = 1
        data.loc[:, name] = tmp
    return data
 
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
 

def process_test(receiver,transpath):

    receiver = log_receiver(receiver)
    transmitter = pd.read_csv(transpath)
    transmitter = log_transmitter(transmitter)
    nearestFeature = get_nearestFeature(receiver, transmitter, top_n_num=5)
    receiver = pd.concat([receiver, nearestFeature], axis=1)
    receiver = add_clutter_type(receiver)
    receiver = receiver.loc[:, ['Azimuth_2', 'Azimuth_3', 'Height_0', 'Azimuth_1', 'Azimuth_4', 'Altitude', 
    'neighbor_num_3000', 'Azimuth_0', 'neighbor_num_4000', 'neighbor_num_2000', 'neighbor_num_5000', 
    'nearest_distance_3', 'nearest_distance_4', 'nerest_Q', 'nearest_distance_2', 'nearest_distance_1', 
    'Y', 'nearest_distance_0', 'X', 'nearest_theta']]
    return receiver

class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        '''
        data is the test data information{'file1': [filename, values]}
        '''
        # print(os.listdir('./'))
        self.xgbt_results = None
        with open(self.model_path + '/lgb.pkl', 'rb') as fw:
            xgb_model = cPickle.load(fw)
        xgbt_results = []
        preprocessed_data = {}
        filesDatas = []
        transpath = self.model_path + '/transmitter.csv'
        for k, v in data.items():
            for file_name, file_content in v.items():
                print(file_name)
                pb_data = pd.read_csv(file_content)
                input_data = np.array(pb_data.get_values()[:,0:17], dtype=np.float32)
                print(file_name, input_data.shape)
                filesDatas.append(input_data)
                predicted = xgb_model.predict(process_test(pb_data,transpath).values)
                # print(predicted)
                xgbt_results.append(predicted)
        self.xgbt_results = xgbt_results

        filesDatas = np.array(filesDatas,dtype=np.float32).reshape(-1, 17)
        preprocessed_data['myInput'] = filesDatas        
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
        return preprocessed_data

    def _postprocess(self, data): 

        infer_output = {"RSRP": []}
        count = 0
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = [[i]for i in self.xgbt_results[count]]
            count += 1
        return infer_output

