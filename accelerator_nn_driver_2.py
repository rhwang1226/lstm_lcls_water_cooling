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

combined_data = np.concatenate((data1, data2), axis=1)


WGHPWR = 6.2
wghpwr_arr = np.full((combined_data.shape[0], 1), WGHPWR)

LCW_SUPPLY_TEMP = 32
lcw_supply_temp_arr = np.full((combined_data.shape[0], 1), LCW_SUPPLY_TEMP)

combined_data = np.concatenate((combined_data, wghpwr_arr), axis=1)
combined_data = np.concatenate((combined_data, lcw_supply_temp_arr), axis=1)

'''
'WGTCAVT','WGTCAV','WGFCTLT','WGFCTL','WGFLWT','WGFLW','GCVFPMT','GCVFPM'
this is: 
#1. the time for the Cavity temp
#2. the Cavity temp
#3. the time for the Flow Valve
#4. the Flow Valve
#5. the time for the Flow Rate
#6. the Flow Rate
#7. the gun forward power time
#8. the gun forward power
'''

training_set_x = combined_data[:, [3, 5, 7, 8, 9]]
training_set_y = combined_data[:, [1]]

# modify sliding windows so that it takes in datax and datay (numpy arrays)
def sliding_windows(dataX, dataY, seq_length):
    x = np.zeros((0, seq_length, 5))
    y = np.zeros((0, 1, 1))

    _x = np.zeros((1, seq_length, 5))
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

sc_x = MinMaxScaler()
training_data_xS = sc_x.fit_transform(training_set_x)

sc_y = MinMaxScaler()
training_data_yS = sc_y.fit_transform(training_set_y)

seq_length = 4
x, y = sliding_windows(training_data_xS, training_data_yS, seq_length)

train_size = int(np.shape(y)[0] * 0.67)
test_size = int(np.shape(y)[0] - train_size)

dataX = Variable(torch.Tensor(x))
dataY = Variable(torch.Tensor(y))

trainX = Variable(torch.Tensor(x[0:train_size]))
trainY = Variable(torch.Tensor(y[0:train_size]))

testX = Variable(torch.Tensor(x[train_size:len(x)]))
testY = Variable(torch.Tensor(y[train_size:len(y)]))

num_epochs = 2000
learning_rate = 0.01

input_size = 5
hidden_size = 2
num_layers = 1

y_features = 1 #feature size (1 time series)

lstm = lstm_accelerator.LSTM(y_features, input_size, hidden_size, num_layers, seq_length)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY.reshape((trainY.size()[0], 1)))
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


lstm.eval()
train_predict = lstm(dataX)

data_predict = train_predict.data.numpy()

dataY_plot = dataY.data.numpy().reshape((14466,1))

data_predict = sc_y.inverse_transform(data_predict)
dataY_plot = sc_y.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')


# Add the legend with the labels specified above
plt.legend()

plt.show()