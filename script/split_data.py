import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

# Load data
df = pd.read_csv(DATA_PATH, encoding="latin1")

# Clean column names
df.columns = df.columns.str.lower()

# =========================
# AREA TYPE MAPPING
# =========================
def area_mapper(x):
    if "Residential" in str(x):
        return "residential"
    elif "Industrial" in str(x):
        return "industrial"
    else:
        return "other"

df["area_type"] = df["type"].apply(area_mapper)

# =========================
# SPLIT DATA
# =========================
residential_df = df[df["area_type"] == "residential"]
industrial_df = df[df["area_type"] == "industrial"]
other_df = df[df["area_type"] == "other"]

# =========================
# SAVE FILES (in data folder)
# =========================
residential_df.to_csv(os.path.join(BASE_DIR, "data", "residential.csv"), index=False)
industrial_df.to_csv(os.path.join(BASE_DIR, "data", "industrial.csv"), index=False)
other_df.to_csv(os.path.join(BASE_DIR, "data", "other.csv"), index=False)

print("Data split completed successfully!")