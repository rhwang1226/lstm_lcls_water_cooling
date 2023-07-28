import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Load the data
data = pd.read_csv("airline-passengers.csv")

# Extract the "Passengers" column
passengers_data = data["Passengers"].values.reshape(-1, 1)

# Normalize the data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_passengers = scaler.fit_transform(passengers_data)

# Define the size of the training set (80% of the data)
train_size = int(len(normalized_passengers) * 0.8)
train_data = normalized_passengers[:train_size]
test_data = normalized_passengers[train_size:]

def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i+time_steps, 0])
        y.append(dataset[i+time_steps, 0])
    return np.array(X), np.array(y)

# Define the number of time steps for the LSTM
time_steps = 10

# Create training and testing datasets
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Reshape the input data to fit the LSTM model (samples, time_steps, features)
X_train = X_train.view(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.view(X_test.shape[0], X_test.shape[1], 1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define model parameters
input_size = 1
hidden_size = 50
num_layers = 2

# Create the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 16

for epoch in range(num_epochs):
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

train_predictions = scaler.inverse_transform(train_predictions.numpy())
y_train = scaler.inverse_transform(y_train.reshape(-1, 1).numpy())
test_predictions = scaler.inverse_transform(test_predictions.numpy())
y_test = scaler.inverse_transform(y_test.reshape(-1, 1).numpy())

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data["Month"][:train_size], y_train, label='Actual (Training)')
plt.plot(data["Month"][train_size+time_steps:], y_test, label='Actual (Testing)')
plt.plot(data["Month"][time_steps:train_size+time_steps], train_predictions, label='Predicted (Training)')
plt.plot(data["Month"][train_size+2*time_steps:], test_predictions, label='Predicted (Testing)')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.show()