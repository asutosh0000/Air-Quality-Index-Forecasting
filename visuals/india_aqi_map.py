import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load India shapefile (download separately)


india_map = gpd.read_file("/Users/asutoshparija/Downloads/Programming/Python/Air Quality Index Predicting/visuals/states_india copy.geojson")

# print(india_map.columns)
# Example AQI data (replace with your predicted values)
data = {
    "state": ["Odisha", "Maharashtra", "Delhi", "Karnataka"],
    "AQI": [80, 120, 250, 60]
}

df = pd.DataFrame(data)

# Merge map + data
merged = india_map.merge(df, left_on="NAME_1", right_on="state", how="left")
merged["AQI"] = merged["AQI"].fillna(50)   # default value
# Plot
plt.figure(figsize=(8,8))
merged.plot(column="AQI", cmap="RdYlGn_r", legend=True, missing_kwds={
    "color": "lightgrey",
    "label": "No Data"
})
plt.title("Predicted AQI Across India")
plt.axis("off")

plt.savefig("india_aqi_map.png")
plt.show()