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


def im2col(x):
    # Parameters
    M,N = x.shape
    col_extent = N - 3 + 1
    row_extent = M - 3 + 1
    
    start_idx = np.arange(3)[:,None]*N + np.arange(3)
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    return np.take (x,start_idx.ravel()[:,None] + offset_idx.ravel()[::1])


def conv(x, w_conv, b_conv):
    # TO DO
    num_out_ch = b_conv.shape[0]
    num_in_ch = x.shape[2]
    h_input = x.shape[0]
    w_input = x.shape[1]
    h_filter = w_conv.shape[0]
    w_filter = w_conv.shape[1]
    y = np.zeros([h_input, w_input, num_out_ch])

    # Zero Padding
    padding_h = int(h_filter / 2)
    padding_w = int(w_filter / 2)
    x_zp = np.empty([h_input + 2 * padding_h, w_input + 2 * padding_w, num_in_ch])
    for i in range(num_in_ch):
        x_zp[:, :, i] = np.pad(x[:, :, i], ((padding_h, padding_h), (padding_w, padding_w)), 'constant', constant_values=0)

    # Convolution
    # for n_out_ch in range(num_out_ch):
    #     for i in range(h_input):
    #         for j in range(w_input):
    #             for k in range(h_filter):
    #                 for l in range(w_filter):
    #                     for n_in_ch in range(num_in_ch):
    #                         y[i][j] += x_zp[i + k][j + l] * w_conv[h_filter - k - 1][w_filter - l -1][n_in_ch][n_out_ch]

    for n_out_ch in range(num_out_ch):
        for n_in_ch in range(num_in_ch):
            row = np.flip(w_conv[:, :, n_in_ch, n_out_ch].flatten())
            column = im2col(x_zp[:, :, n_in_ch])
            y[:, :, n_out_ch] = (row @ column).reshape([h_input, w_input])
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    num_out_ch = b_conv.shape[0]
    num_in_ch = x.shape[2]
    h_input = x.shape[0]
    w_input = x.shape[1]
    h_filter = w_conv.shape[0]
    w_filter = w_conv.shape[1]
    
    dl_dw = np.zeros([3, 3, 1, 3])
    for n_out_ch in range(num_out_ch):
        for hf in range(h_filter):
            for wf in range(w_filter):
                for n_in_ch in range(num_in_ch):
                    for hi in range(h_input - h_filter):
                        for wi in range(w_input - w_filter):
                            dl_dw[hf][wf][n_in_ch][n_out_ch] += dl_dy[hi][wi][n_out_ch] * x[hi + hf][wi + wf][n_in_ch]
    
    dl_db = dl_dy[0][0][:]
    dl_db = dl_db.reshape(dl_db.shape[0], -1)
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    y = np.empty([int(x.shape[0] / 2), int(x.shape[1] / 2), x.shape[2]])
    for i in range(x.shape[2]):
        for j in range(0, x.shape[0], 2):
            for k in range(0, x.shape[1], 2):
                y[int(j / 2)][int(k / 2)][i] = np.max(x[j:j+2, k:k+2, i])
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    dl_dx = np.zeros(x.shape)
    for i in range(x.shape[2]):
        for j in range(0, x.shape[0], 2):
            for k in range(0, x.shape[1], 2):
                dl_dx[j][k][i] = dl_dy[int(j / 2)][int(k / 2)][i]
    return dl_dx


def flattening(x):
    # TO DO
    y = x.flatten()
    y = y.reshape(y.shape[0], -1)
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape)
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
    learning_rate = 0.1
    decay_rate = 0.9
    w_conv = np.random.normal(0, 0.1, size=(3, 3, 1, 3))
    b_conv = np.random.normal(0, 0.1, size=(3, 1))
    w_fc = np.random.normal(0, 0.1, size=(10, 147))
    b_fc = np.random.normal(0, 0.1, size=(10, 1))
    loss = []

    k = 0
    batch_size = 0
    for i in range(5000):
        print(i)
        if ((i % 1000) == 0) and ((i / 1000) > 0):
            learning_rate *= decay_rate
        sum_dL_dw_conv = 0
        sum_dL_db_conv = 0
        sum_dL_dw_fc = 0
        sum_dL_db_fc = 0
        batch_size = mini_batch_x[k].shape[1]
        sum_loss = 0

        for j in range(batch_size):
            x = mini_batch_x[k][:, j]
            x = x.reshape(14, 14, 1)

            # Convolutional Layer
            ac = conv(x, w_conv, b_conv)

            # Relu
            f = relu(ac)

            # Max Pooling
            mp = pool2x2(f)

            # Flatten
            fln = flattening(mp)

            # FC
            afc = fc(fln, w_fc, b_fc)

            # Soft Max and Loss Function
            y_gt = mini_batch_y[k][:, j]
            y_gt = y_gt.reshape(y_gt.shape[0], -1)
            l, dl_dy = loss_cross_entropy_softmax(afc, y_gt)
            sum_loss += l 

            # FC Backward
            dl_dx, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, fln, w_fc, b_fc, afc)

            # Flatten Backward
            dl_dx = flattening_backward(dl_dx, mp, fln)

            # Max Pooling Backward
            dl_dx = pool2x2_backward(dl_dx, f, mp)

            # Relu Backward
            dl_dx = relu_backward(dl_dx, ac, f)

            # Convolutional Layer Backward
            dl_dw_conv, dl_db_conv = conv_backward(dl_dx, x, w_conv, b_conv, ac)

            sum_dL_dw_conv += dl_dw_conv
            sum_dL_db_conv += dl_db_conv
            sum_dL_dw_fc += dl_dw_fc
            sum_dL_db_fc += dl_db_fc

        k = (k + 1) % len(mini_batch_x)
        
        w_conv -= (learning_rate / batch_size) * sum_dL_dw_conv
        b_conv -= (learning_rate / batch_size) * sum_dL_db_conv
        w_fc -= (learning_rate / batch_size) * sum_dL_dw_fc
        b_fc -= (learning_rate / batch_size) * sum_dL_db_fc
        
        loss.append(sum_loss / batch_size)

    # x = mini_batch_x[0][:, 0]
    # x = x.reshape(14, 14, 1)
    # conv(x, w_conv, b_conv)
    # y = pool2x2(x[4:8, 4:8, :])
    # dl = pool2x2_backward(y, x[4:8, 4:8, :], y)
    # yf = flattening(y)
    # yr = flattening_backward(yf, y, y)
    
    plt.plot(loss)
    plt.show()

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    # main.main_mlp()
    main.main_cnn()



