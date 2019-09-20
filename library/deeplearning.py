import tensorflow as tf
import numpy as np

conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d

def create_var(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)

def ReLUBN(var):
    return tf.nn.relu(tf.layers.batch_normalization(var))

def chunkBigData(feed_dict, batchsize):
    num = int(feed_dict[list(feed_dict.keys())[0]].shape[0]/batchsize)+1
    for i in range(num):
        feedtmp = {key: feed_dict[key][i*batchsize: (i+1)*batchsize] for key in list(feed_dict.keys())}
        yield len(feedtmp[list(feedtmp.keys())[0]]), feedtmp

def evaluate(sess, feed_dict, batchsize, condition, evalList):
    board = []
    for length, feedtmp in chunkBigData(feed_dict, batchsize):
        if length != 0:
            feedtmp[condition] = 0  # Add condition key
            loss_ = sess.run(evalList, feed_dict=feedtmp)
            targets_ = feedtmp[list(feedtmp.keys())[-2]]  # get the value of targets one-hot
            board.append([length, length*loss_])
    board = np.array(board)
    loss_ = np.sum(board[:, 1])/np.sum(board[:, 0])
    return loss_, np.sqrt(loss_)


def Image_ConvNet(inputs):
    conv1 = tf.layers.conv2d(
                    inputs=inputs,
                    filters=8,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
                    inputs=ReLUBN(conv1), 
                    pool_size=[4, 4], 
                    strides=4)
    conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=8,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
                    inputs=ReLUBN(conv2), 
                    pool_size=[2, 2], 
                    strides=2)
    state = tf.layers.dense(inputs=tf.layers.flatten(pool2), units=128, activation=tf.nn.relu)
    return state

def MLP(inputs):
    mlp1 = tf.layers.dense(inputs=inputs, units=32)
    mlp1 = tf.layers.dropout(ReLUBN(mlp1), 0.2)
    return mlp1

