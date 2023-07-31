import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

import accel_data_prep
import lstm_accelerator

data1 = accel_data_prep.get_guntest()
data2 = accel_data_prep.get_gunpower()

# modify sliding windows so that it takes in datax and datay (numpy arrays)
def sliding_windows(dataX, dataY, seq_length):
    x = np.zeros((0, seq_length, 4))
    y = np.zeros((0, 1, 1))

    _x = np.zeros((1, seq_length, 4))
    _y = np.zeros((1, 1, 1))

    print("np.shape(dataX)[0]: ", np.shape(dataX)[0])
    for i in range(np.shape(dataX)[0] - seq_length):
        _x[0] = dataX[i : (i + seq_length),:] #this will become datax
        _y[0] = dataY[i + seq_length, 0]
        x = np.concatenate((x,_x),axis=0)
        y = np.concatenate((y,_y),axis=0)
        print("_x[0]: ", _x[0])
        print("_y[0]: ", _y[0])

    print(x.shape)
    print(y.shape)
    return x, y

