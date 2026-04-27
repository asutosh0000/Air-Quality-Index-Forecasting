import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

modes = np.load(os.path.join(DATA_DIR, "vmd_modes.npy"))

SEQ_LEN = 10
X, y = [], []

for i in range(modes.shape[1] - SEQ_LEN):
    X.append(modes[:, i:i+SEQ_LEN].T)
    y.append(modes[0, i+SEQ_LEN])  # predict main mode

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

np.save(os.path.join(DATA_DIR, "X_vmd.npy"), X)
np.save(os.path.join(DATA_DIR, "y_vmd.npy"), y)

print("VMD sequences ready")