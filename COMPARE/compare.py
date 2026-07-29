import matplotlib
matplotlib.use('Agg') # Ensures no GUI window is opened, avoiding system errors
import matplotlib.pyplot as plt
import numpy as np

# 1. ACTUAL DATA FROM YOUR UPLOADED FILES
models = ['MobileNetV2', 'ResNet50', 'ViT', 'DenseNet', 'Simple CNN', 'SVM']
# Validation accuracies extracted from your PDF/Image results
val_accuracies = [82.22, 90.00, 99.00, 70.00, 80.66, 58.00]
# Training accuracies (DenseNet set to 92.24%)
train_accuracies = [89.50, 95.00, 91.00, 92.24, 72.00, 60.00]

# Resource Metadata (Params in Millions and Size in MB)
params_m = [2.26, 23.5, 86.0, 7.0, 1.0, 0.05]
model_sizes_mb = [8.6, 98.98, 340.0, 26.87, 4.0, 0.5]
colors = ['#28B463', '#2E86C1', '#9B59B6', '#F39C12', '#E74C3C', '#95A5A6']

# --- GRAPH 1: VALIDATION ACCURACY ---
plt.figure(figsize=(12, 7))
bars = plt.bar(models, val_accuracies, color=colors, edgecolor='black', alpha=0.85)
plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Comparison of Model Validation Accuracies\n(Actual Results from Project Files)', fontsize=14, fontweight='bold', pad=20)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('actual_validation_accuracy.png', dpi=300)
print("Saved: actual_validation_accuracy.png")

# --- GRAPH 2: TRAINING vs VALIDATION (OVERFITTING ANALYSIS) ---
plt.figure(figsize=(12, 7))
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy', color='#1F4E79', alpha=0.7)
plt.bar(x + width/2, val_accuracies, width, label='Validation Accuracy', color='#E46C0A', alpha=0.9)

plt.ylabel('Accuracy (%)', fontweight='bold')
plt.xticks(x, models, fontweight='bold')
plt.title('Training vs. Validation Accuracy Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.5)

# Adding a note about DenseNet's gap
plt.annotate('High Gap: Potential Overfitting', xy=(3.1, 75), xytext=(4, 90),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
plt.savefig('train_val_gap.png', dpi=300)
print("Saved: train_val_gap.png")

# --- GRAPH 3: EFFICIENCY FRONTIER ---
plt.figure(figsize=(11, 6))
for i in range(len(models)):
    # Bubble size represents physical model size on disk
    plt.scatter(params_m[i], val_accuracies[i], s=np.sqrt(model_sizes_mb[i])*150, 
                color=colors[i], label=models[i], edgecolors='black', alpha=0.7)
    plt.text(params_m[i], val_accuracies[i] + 3, models[i], fontsize=10, fontweight='bold', ha='center')

plt.xscale('log')
plt.xlabel('Model Complexity (Millions of Parameters - Log Scale)', fontweight='bold')
plt.ylabel('Validation Accuracy (%)', fontweight='bold')
plt.title('Accuracy vs. Structural Complexity Frontier', fontweight='bold')
plt.grid(True, which="both", linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('actual_efficiency_frontier.png', dpi=300)
print("Saved: actual_efficiency_frontier.png")