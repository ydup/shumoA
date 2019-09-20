'''
Masked output, pix2pix model
'''
import tensorflow as tf
from library.deeplearning import *
from library.util import *
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from library.feature import *
import copy

Image, Mask, Idx2d, Const, Target = [], [], [], [], []
pp = preprocess()
FileNames = os.listdir('../../Downloads/train_set/')
for name in tqdm(FileNames[0: 100]):
    data = pd.read_csv('../../Downloads/train_set/{0}'.format(name))
    data = add_feature(data)
    image, mask, idx2d, const = pp.get_image_const(data)
    target = pp.get_target(data)
    Image.append(image)
    Mask.append(mask)
    Const.append(const)
    Target.append(target)

Image = np.stack(Image, axis=0)
Mask = np.array(np.stack(Mask, axis=0), dtype=np.bool)
Const = np.stack(Const, axis=0)
Target = np.stack(Target, axis=0)

minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
minmax.fit(Const)
Const = minmax.transform(Const)

# Parameters setting
batch_size = 8

learning_rate = 0.01
epoch = 200

thres = 90
train_input, train_const, train_mask, train_target = Image[0: thres], Const[0: thres], Mask[0: thres], Target[0: thres]
test_input, test_const, test_mask, test_target = Image[thres: ], Const[thres: ], Mask[thres: ], Target[thres: ]

trainImageshape = list(train_input.shape)
trainConstshape = list(train_const.shape)
trainMaskshape = list(train_mask.shape)
trainTargetshape = list(train_target.shape)

def placeholder(ImageShape, ConstShape, MaskShape, TargetShape):
    '''
    Placeholder
    '''
    image_shape, const_shape, mask_shape, target_shape = copy.copy(ImageShape), copy.copy(ConstShape), copy.copy(MaskShape), copy.copy(TargetShape)
    image_shape[0], const_shape[0], mask_shape[0], target_shape[0] = None, None, None, None
    inputs_image = tf.placeholder(shape=image_shape, dtype=tf.float32, name='inputs_image')
    inputs_const = tf.placeholder(shape=const_shape, dtype=tf.float32, name='inputs_const')
    inputs_mask = tf.placeholder(shape=mask_shape, dtype=tf.bool, name='inputs_image')
    targets_image = tf.placeholder(shape=target_shape, dtype=tf.float32, name='target')  # placeholder for target
    return inputs_image, inputs_const, inputs_mask, targets_image

tf.reset_default_graph()  # reset the graph

with tf.name_scope('Placeholder'):
    inputs_image = tf.placeholder(shape=trainImageshape, dtype=tf.float32, name='inputs_image')
    inputs_const = tf.placeholder(shape=trainConstshape, dtype=tf.float32, name='inputs_const')
    inputs_mask = tf.placeholder(shape=trainMaskshape, dtype=tf.bool, name='inputs_image')
    targets_image = tf.placeholder(shape=trainTargetshape, dtype=tf.float32, name='target')  # placeholder for target
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs_image, inputs_const, inputs_mask, targets_image))
    dataset = dataset.batch(batch_size).repeat()
    iterator = dataset.make_initializable_iterator()
    # Condition to choose get_next or the placeholder()
    condition = tf.placeholder(tf.int32, shape=[], name="condition")
    images, consts, masks, targets_image = tf.cond(condition > 0, 
                                            lambda: iterator.get_next(), 
                                            lambda: placeholder(trainImageshape, trainConstshape, trainMaskshape, trainTargetshape))

with tf.name_scope('Convolution'):
    imageState = Image_ConvNet(images)  # (batch, 128)
with tf.name_scope('MLP'):
    constState = MLP(consts)  # (batch, 32)
with tf.name_scope('Concat'):
    state = tf.concat([imageState, constState], axis=1)  # (batch, 128+32)
    state = tf.expand_dims(tf.expand_dims(state, axis=1), axis=1)
with tf.name_scope('De-convolution'):
    kernel1 = create_var("kernel", 
                        [8, 8, 1, int(state.get_shape()[-1])], conv2d_initializer())
    strides = [1, 2, 2, 1]
    deconv = tf.nn.conv2d_transpose(state, kernel1, 
                output_shape=[int(inputs_image.get_shape()[0]), int(inputs_image.get_shape()[1]), int(inputs_image.get_shape()[2]), 1], 
                strides=strides, padding='SAME')
    deconv = ReLUBN(deconv)

with tf.name_scope('Masker'):
    masked_deconv = tf.boolean_mask(deconv, masks)
    masked_target = tf.boolean_mask(tf.squeeze(targets_image, axis=3), masks)

with tf.name_scope('compile'):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(masked_deconv, masked_target))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

n_batches = train_input.shape[0] // batch_size + 1

with tf.Session() as sess:
    # train_input, train_const, train_mask, train_target
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={images:train_input, 
                                                consts:train_const, 
                                                masks:train_mask, 
                                                targets_image:train_target})
    for ep in range(epoch):
        for _ in tqdm(range(n_batches), desc='Epoch: {0}'.format(ep)):
            sess.run(train_step, feed_dict={condition: 1})  # train the NN with enable get_next of the iterator
        loss_, rmse_ = evaluate(sess, {images:train_input, consts:train_const, masks:train_mask, targets_image:train_target}, 32, condition, loss)
        print('Train: Loss={0}\tRMSE={1}'.format(*[loss_, rmse_]))
        loss_, rmse_ = evaluate(sess, {images:test_input, consts:test_const, masks:test_mask, targets_image:test_target}, 32, condition, loss)
        print('Test: Loss={0}\tRMSE={1}'.format(*[loss_, rmse_]))






    


