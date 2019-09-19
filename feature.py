from math import atan


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
    data.loc[:, 'seperation_distance'] = np.sqrt((data.loc[:, 'X'] - data.loc[:, 'Cell X'])**2 + (data.loc[:, 'Y'] - data.loc[:, 'Cell Y'])**2)
    data.loc[:, 'dh'] = np.log10(1+data.loc[:, 'seperation_distance'])*np.log10(1+data.loc[:, 'Effective_Cell_Height'])
    data.loc[:, 'X_relative'] = data.loc[:, 'X'] - data.loc[:, 'Cell X']
    data.loc[:, 'Y_relative'] = data.loc[:, 'Y'] - data.loc[:, 'Cell Y']
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


