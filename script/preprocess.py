import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
# 👉 NEW: scripts folder path
SCRIPT_DIR = os.path.dirname(__file__)

df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = df.columns.str.lower()

# Clean
# df = df.dropna()

df_numeric = df.select_dtypes(include=["number"])
df_numeric = df_numeric.fillna(df_numeric.mean())
df[df_numeric.columns] = df_numeric
# Features & target
target = "pm2_5"
features = df.select_dtypes(include=["number"]).columns.tolist()
features.remove(target)

X = df[features]
y = df[target]
y = y.values.reshape(-1, 1)
y = MinMaxScaler().fit_transform(y).flatten()
# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_regression, k=6)
X_selected = selector.fit_transform(X_scaled, y)

# =========================
# SAVE IN scripts FOLDER
# =========================
np.save(os.path.join(DATA_DIR, "X_processed.npy"), X_selected)
np.save(os.path.join(DATA_DIR, "y_processed.npy"), y)

print("Preprocessed data saved in scripts folder")