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
        batch_y = np.empty([10, 0])
        for i in range(index, index + batch_size):
            sample = im_train[:, order[i]].reshape(196, -1)
            batch_x = np.append(batch_x, sample, axis = 1)
            
            sample_label = np.empty([10, 1])
            for j in range(10):
                if j == label_train[0][order[i]]:
                    sample_label[j] = 1
                else:
                    sample_label[j] = 0
            batch_y = np.append(batch_y, sample_label, axis = 1)
        index += batch_size
        mini_batch_x.append(batch_x)
        mini_batch_y.append(batch_y)

    batch_x = np.empty([196, 0])
    batch_y = np.empty([10, 0])
    for i in range(index, data_num):
            sample = im_train[:, order[i]].reshape(196, -1)
            batch_x = np.append(batch_x, sample, axis = 1)
            
            sample_label = np.empty([10, 1])
            for j in range(10):
                if j == label_train[0][order[i]]:
                    sample_label[j] = 1
                else:
                    sample_label[j] = 0
            batch_y = np.append(batch_y, sample_label, axis = 1)
    mini_batch_x.append(batch_x)
    mini_batch_y.append(batch_y)
                    
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    y = w @ x + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = w.T @ dl_dy
    dl_dw = dl_dy @ x.T
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    dl_dy = y_tilde - y
    l = sum(dl_dy ** 2)
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    y_tilde = np.exp(x)
    y_sum = sum(y_tilde)
    y_tilde /= y_sum
    l = -sum(y * np.log(y_tilde))
    dl_dy = (y_tilde - y)
    return l, dl_dy

def relu(x):
    # TO DO
    y = np.maximum(0, x)
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = np.where(y > 0, dl_dy, 0)
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
    learning_rate = 0.15
    decay_rate = 0.3
    w = np.random.normal(0, 0.1, size=(10, 196))
    b = np.random.normal(0, 0.1, size=(10, 1))
    loss = []
    
    k = 0
    batch_size = 0
    for i in range(8000):
        if ((i % 1000) == 0) and ((i / 1000) > 0):
            learning_rate *= decay_rate
        sum_dL_dw = 0
        sum_dL_db = 0
        batch_size = mini_batch_x[k].shape[1]
        sum_loss = 0
        for j in range(batch_size):
            x = mini_batch_x[k][:, j]
            x = x.reshape(x.shape[0], -1)
            y_predict = fc(x, w, b)
            
            y_gt = mini_batch_y[k][:, j]
            y_gt = y_gt.reshape(y_gt.shape[0], -1)
           
            l, dl_dy = loss_euclidean(y_predict, y_gt)
            sum_loss += l
            _, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_predict)
            sum_dL_dw += dl_dw
            sum_dL_db += dl_db

        k = (k + 1) % len(mini_batch_x)
        w -= (learning_rate / batch_size) * sum_dL_dw
        b -= (learning_rate / batch_size) * sum_dL_db
        loss.append(sum_loss / batch_size)
    
    # x = np.array([1, 2, 3]).reshape(3, -1)
    # w = np.array([[1, 2, 3], [4, 5, 6]])
    # b = np.array([1, 2]).reshape(2, -1)
    # y = fc(x, w, b)
    # l, dl_dy = loss_euclidean(y.reshape(-1), [10, 30])
    # _, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
    # plt.plot(loss)
    # plt.show()
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 0.1
    decay_rate = 0.9
    w = np.random.normal(0, 0.1, size=(10, 196))
    b = np.random.normal(0, 0.1, size=(10, 1))
    loss = []
    
    k = 0
    batch_size = 0
    for i in range(8000):
        if ((i % 1000) == 0) and ((i / 1000) > 0):
            learning_rate *= decay_rate
        sum_dL_dw = 0
        sum_dL_db = 0
        batch_size = mini_batch_x[k].shape[1]
        sum_loss = 0
        for j in range(batch_size):
            x = mini_batch_x[k][:, j]
            x = x.reshape(x.shape[0], -1)
            a = fc(x, w, b)
            
            y_gt = mini_batch_y[k][:, j]
            y_gt = y_gt.reshape(y_gt.shape[0], -1)
            
            l, dl_dy = loss_cross_entropy_softmax(a, y_gt)
            sum_loss += l
            _, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, a)
            sum_dL_dw += dl_dw
            sum_dL_db += dl_db

        k = (k + 1) % len(mini_batch_x)
        w -= (learning_rate / batch_size) * sum_dL_dw
        b -= (learning_rate / batch_size) * sum_dL_db
        loss.append(sum_loss / batch_size)
    
    # plt.plot(loss)
    # plt.show()

    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    learning_rate = 0.1
    decay_rate = 0.9
    w1 = np.random.normal(0, 0.1, size=(30, 196))
    b1 = np.random.normal(0, 0.1, size=(30, 1))
    w2 = np.random.normal(0, 0.1, size=(10, 30))
    b2 = np.random.normal(0, 0.1, size=(10, 1))
    loss = []

    k = 0
    batch_size = 0
    for i in range(8000):
        if ((i % 1000) == 0) and ((i / 1000) > 0):
            learning_rate *= decay_rate
        sum_dL_dw1 = 0
        sum_dL_db1 = 0
        sum_dL_dw2 = 0
        sum_dL_db2 = 0
        batch_size = mini_batch_x[k].shape[1]
        sum_loss = 0

        for j in range(batch_size):
            x = mini_batch_x[k][:, j]
            x = x.reshape(x.shape[0], -1)
            
            # First Layer Forward
            a1 = fc(x, w1, b1)
            f1 = relu(a1)
            
            # Second Layer Forward
            a2 = fc(f1, w2, b2)

            # Soft Max and Loss Function
            y_gt = mini_batch_y[k][:, j]
            y_gt = y_gt.reshape(y_gt.shape[0], -1)
            l, dl_dy = loss_cross_entropy_softmax(a2, y_gt)
            sum_loss += l
            
            # Second Layer Backward
            dl_dx, dl_dw2, dl_db2 = fc_backward(dl_dy, f1, w2, b2, a2)

            # First Layer Backward
            dl_dx = relu_backward(dl_dx, a1, f1)
            _, dl_dw1, dl_db1 = fc_backward(dl_dx, x, w1, b1, a1)
            
            sum_dL_dw1 += dl_dw1
            sum_dL_db1 += dl_db1
            sum_dL_dw2 += dl_dw2
            sum_dL_db2 += dl_db2

        k = (k + 1) % len(mini_batch_x)
        
        w1 -= (learning_rate / batch_size) * sum_dL_dw1
        b1 -= (learning_rate / batch_size) * sum_dL_db1
        w2 -= (learning_rate / batch_size) * sum_dL_dw2
        b2 -= (learning_rate / batch_size) * sum_dL_db2
        
        loss.append(sum_loss / batch_size)
    
    # plt.plot(loss)
    # plt.show()

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    main.main_mlp()
    main.main_cnn()



