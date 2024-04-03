import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    data_num = label_train.shape[1]
    # Shuffle the order
    order = np.random.choice(range(data_num), data_num, replace=False)
    
    index = 0
    mini_batch_x = []
    mini_batch_y = []
    
    while ((data_num - index) > batch_size):
        batch_x = np.empty([196, 0])
        batch_y = np.empty([10, 1])
        for i in range(index, index + batch_size):
            sample = im_train[:, order[i]].reshape(196, -1)
            batch_x = np.append(batch_x, sample, axis = 1)
            for j in range(10):
                if j == label_train[0][order[i]]:
                    batch_y[j] = 1
                else:
                    batch_y[j] = 0
        index += batch_size
        mini_batch_x.append(batch_x)
        mini_batch_y.append(batch_y)

    batch_x = np.empty([196, 0])
    batch_y = np.empty([10, 1])
    for i in range(index, data_num):
            sample = im_train[:, order[i]].reshape(196, -1)
            batch_x = np.append(batch_x, sample, axis = 1)
            for j in range(10):
                if j == label_train[0][order[i]]:
                    batch_y[j] = 1
                else:
                    batch_y[j] = 0
    mini_batch_x.append(batch_x)
    mini_batch_y.append(batch_y)
                    
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    return l, dl_dy

def relu(x):
    # TO DO
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    # TO DO
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



