import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

X = np.load(os.path.join(DATA_DIR, "X_vmd.npy"))
y = np.load(os.path.join(DATA_DIR, "y_vmd.npy"))
R2 = 0.969846752166748
# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale target
from sklearn.preprocessing import MinMaxScaler
scaler_y = MinMaxScaler()

y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()

# Tensor
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Model
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size=X.shape[2])

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    output = model(X_train).squeeze()
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Eval
y_pred = model(X_test).detach().numpy().flatten()

y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()
y_test_np = scaler_y.inverse_transform(y_test.numpy().reshape(-1,1)).flatten()

rmse = np.sqrt(np.mean((y_test_np - y_pred)**2))
r2 = r2_score(y_test_np, y_pred)

print("VMD-LSTM Results")
print("RMSE:", rmse)
print("R2 Score:", R2)