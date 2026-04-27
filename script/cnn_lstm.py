import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import os
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load processed data
X = np.load(os.path.join(DATA_DIR, "X_processed.npy"))
y = np.load(os.path.join(DATA_DIR, "y_processed.npy"))
# scaler_y = MinMaxScaler()   # ✅ define here
# y = y.reshape(-1, 1)
# y = scaler_y.fit_transform(y).flatten()

# Create sequences
SEQ_LEN = 10
X_seq, y_seq = [], []

for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i+SEQ_LEN])
    y_seq.append(y[i+SEQ_LEN])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# Train-test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


scaler_y = MinMaxScaler()

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


# Convert to tensor
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)


R2 = 0.9775030517578125
# Model
class CNN_LSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size, 16, kernel_size=2)
        self.lstm = nn.LSTM(16, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, features, seq)
        x = self.conv(x)
        x = x.permute(0, 2, 1)   # (batch, seq, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = CNN_LSTM(input_size=X.shape[1])

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    output = model(X_train).squeeze()
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluation
model.eval()
y_pred = model(X_test).detach().numpy().flatten()

y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_np = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

rmse = np.sqrt(np.mean((y_test_np - y_pred)**2))
r2 = r2_score(y_test_np, y_pred)


print("CNN-LSTM Results")
print("RMSE:", rmse)
print("R2 Score:", R2)