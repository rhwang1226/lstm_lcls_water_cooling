import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

training_set = pd.read_csv('airline-passengers.csv')

training_set_x = training_set.iloc[:,2:6].values

training_set_y = training_set.iloc[:,1:2].values

#have separate training sets for x and y, x being the 5 time series and y being the one im using for the predictions

#plt.plot(training_set_x, label = 'Airline Passangers Data')
#plt.show()

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


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

num_epochs = 2000
learning_rate = 0.01

input_size = 4
hidden_size = 2
num_layers = 1

y_features = 1 #feature size (1 time series)

lstm = LSTM(y_features, input_size, hidden_size, num_layers)

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

dataY_plot = dataY.data.numpy().reshape((140,1))

data_predict = sc_y.inverse_transform(data_predict)
dataY_plot = sc_y.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')


# Add the legend with the labels specified above
plt.legend()

plt.show()
