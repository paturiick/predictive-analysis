import streamlit as st
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# LSTM Model Definition
class DisasterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DisasterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Data preprocessing and sequence creation
def preprocess_and_create_sequences(data, seq_length):
    data = data[['Start Year','Start Month','Start Day','End Year','End Month','End Day','No. Affected', 'Disaster Type', 
                 'Total Deaths', 'No. Injured', 'Total Damage (\'000 US$)', 'Total Affected']].fillna(0)
    data['Disaster Type'] = data['Disaster Type'].astype('category').cat.codes
    data = data.groupby('Start Year').sum().reset_index()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.drop('Start Year', axis=1))

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length, :-1])
        y.append(scaled_data[i+seq_length, -1])
    return np.array(X), np.array(y), scaler

# Train and evaluate the LSTM model
def train_lstm(X_train, y_train, X_test, y_test, seq_length, input_size, hidden_size, num_layers, output_size, num_epochs, batch_size):
    model = DisasterLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    seq_length = torch.tensor(seq_length, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        st.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / (len(X_train) / batch_size):.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        test_loss = criterion(predictions, y_test).item()
    
    return model, predictions, test_loss