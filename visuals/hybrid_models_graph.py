import matplotlib.pyplot as plt

# Models and R2 values
models = [
    "VMD-LSTM",
    "CNN-LSTM",
    "CNN-BiLSTM-Attention",
    "ConvLSTM"
]

r2_values = [
    0.969846752166748,
    0.9775030517578125,
    0.9825030517578125,
    0.98709204101562
]

# Plot
plt.figure(figsize=(8,6))
bars = plt.bar(models, r2_values)

# Add values on top
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y, f"{y:.3f}",
             ha='center', va='bottom')

plt.title("Hybrid Models Comparison (R2 Score)")
plt.xlabel("Models")
plt.ylabel("R2 Score")

# Save graph
plt.savefig("visuals/hybrid_models_comparison.png")

plt.show()