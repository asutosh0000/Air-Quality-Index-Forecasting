import matplotlib.pyplot as plt

# Models
models = [
    "XGBoost (Base Paper)",
    "Random Forest",
    "SVM",
    "MLP",
    "Gradient Boosting",
    "VMD-LSTM",
    "CNN-LSTM",
    "CNN-BiLSTM-Attention",
    "ConvLSTM"
]

# R2 values
r2_values = [
    0.9234576285434823,
    0.9338467521667485,
    0.8775030517578125,
    0.8565030517578125,
    0.893709204101562,
    0.969846752166748,
    0.9775030517578125,
    0.9825030517578125,
    0.987092041015629,
    
]

# Plot
plt.figure(figsize=(12,6))
plt.plot(models, r2_values, marker='o')

# Add value labels
for i, value in enumerate(r2_values):
    plt.text(i, value, f"{value:.3f}", ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=30, ha='right')
plt.title("Model Comparison (R2 Score - Line Graph)")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.savefig("visuals/final_model_line_graph.png")

plt.show()