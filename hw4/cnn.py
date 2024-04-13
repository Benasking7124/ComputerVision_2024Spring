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


def im2col(img, block_size = (3, 3) ,step_size = 1):
    row, column = img.shape
    col_extent = column - block_size[1] + 1
    row_extent = row - block_size[0] + 1
    
    # Index of the first block
    first_batch = np.arange(block_size[0])[:, None] * column + np.arange(block_size[1])
    
    # Offset from first block to the remaining blocks
    offset_index = (np.arange(0, row_extent, step_size)[:, None] * column + np.arange(0, col_extent, step_size)).ravel()
    
    # Index of every element in the col matrix, number of row of index_matrix will be (block_size * block size)
    index_matrix = first_batch.ravel()[:,None] + offset_index
    
    return np.take(img, index_matrix)


def conv(x, w_conv, b_conv):   # The implemetation will actually be correlation
    # TO DO
    num_out_ch = b_conv.shape[0]
    h_input, w_input, num_in_ch = x.shape
    h_filter, w_filter, _, _ = w_conv.shape

    # Zero Padding
    padding_h = int(h_filter / 2)
    padding_w = int(w_filter / 2)
    x_zp = np.pad(x, ((padding_h, padding_h), (padding_w, padding_w), (0, 0)), 'constant', constant_values=0)

    # Bias Term
    y = np.tile(b_conv.flatten(), (h_input, w_input, 1))

    # Correlation
    for n_in_ch in range(num_in_ch):
        img_column = im2col(x_zp[:, :, n_in_ch])
        for n_out_ch in range(num_out_ch):
            w_flatten = w_conv[:, :, n_in_ch, n_out_ch].flatten()
            y[:, :, n_out_ch] += (w_flatten @ img_column).reshape([h_input, w_input])

    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    num_out_ch = b_conv.shape[0]
    h_input, w_input, num_in_ch = x.shape
    h_filter, w_filter, _, _ = w_conv.shape
    
    # Zero Padding
    padding_h = int(h_filter / 2)
    padding_w = int(w_filter / 2)
    x_zp = np.pad(x, ((padding_h, padding_h), (padding_w, padding_w), (0, 0)), 'constant', constant_values=0)

    # Correlation between x and dl_dy
    dl_dw = np.empty([h_filter, w_filter, num_in_ch, num_out_ch])
    for n_in_ch in range(num_in_ch):
        img_column = im2col(x_zp[:, :, n_in_ch], (h_input, w_input))
        for n_out_ch in range(num_out_ch):
            dl_dy_flatten = dl_dy[:, :, n_out_ch].flatten()
            dl_dw[:, :, n_in_ch, n_out_ch] = (dl_dy_flatten @ img_column).reshape(h_filter, w_filter)

    # Bias Term
    dl_db = np.sum(dl_dy, axis = (0, 1))[:, None]
    
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
                index_flaten = np.argmax(x[j:j+2, k:k+2, i])
                index_row = int(index_flaten / 2)
                index_column = index_flaten % 2
                dl_dx[j + index_row][k + index_column][i] = dl_dy[int(j / 2)][int(k / 2)][i]
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
    learning_rate = 0.05
    decay_rate = 0.9
    w_conv = np.random.normal(0, 0.1, size=(3, 3, 1, 3))
    b_conv = np.random.normal(0, 0.1, size=(3, 1))
    w_fc = np.random.normal(0, 0.1, size=(10, 147))
    b_fc = np.random.normal(0, 0.1, size=(10, 1))
    loss = []

    k = 0
    batch_size = 0
    for i in range(8000):
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

            # For multichanel input, if the format is img = img1.flatten() + img2.flatten(), the reshape could look like this:
            # x = x.reshape((2, 14, 14), order='F').transpose(1, 2, 0)
            
            x = x.reshape((14, 14, 1), order='F')

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
    
    # plt.plot(loss)
    # plt.show()

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



