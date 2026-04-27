import numpy as np
import os
from vmdpy import VMD

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

y = np.load(os.path.join(DATA_DIR, "y_processed.npy"))

# VMD parameters
alpha = 2000
tau = 0
K = 4        # number of modes
DC = 0
init = 1
tol = 1e-7

u, _, _ = VMD(y, alpha, tau, K, DC, init, tol)

print("Modes shape:", u.shape)

np.save(os.path.join(DATA_DIR, "vmd_modes.npy"), u)