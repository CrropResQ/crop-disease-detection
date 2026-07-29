import matplotlib
# CRITICAL: This line prevents the _tkinter.TclError on Windows
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np

# 1. CONSOLIDATED DATA
# Sorted by "Relevance for Field Deployment" (Most Significant First)
# Data source: Your uploaded PDFs (MobileNet, ResNet, DenseNet, CNN) + benchmarks (ViT, SVM)
models = ['MobileNetV2', 'ViT', 'ResNet50', 'DenseNet', 'Simple CNN', 'SVM']
accuracies = [82.22, 94.50, 91.13, 70.00, 65.00, 58.00]
params_m = [2.26, 86.0, 23.5, 7.0, 1.0, 0.05]   # Parameters in Millions
model_sizes_mb = [8.6, 340.0, 98.98, 26.87, 4.0, 0.5] # Storage in MB

# Distinct colors for your report
colors = ['#28B463', '#9B59B6', '#2E86C1', '#F39C12', '#E74C3C', '#95A5A6']

# --- GRAPH 1: ACCURACY VS RELEVANCE ---
def generate_suitability_chart():
    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.85)

    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architectures (Ranked by Practical Significance)', fontsize=12, fontweight='bold')
    plt.title('Suitability Comparison: Performance Sorted by Relevance', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval}%', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.annotate('Highest Practical Relevance\n(Mobile Ready)', 
                 xy=(0, 82.22), xytext=(0.5, 105),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('suitability_ranking.png', dpi=300)
    print("Saved: suitability_ranking.png")

# --- GRAPH 2: EFFICIENCY ANALYSIS (Accuracy vs Complexity) ---
def generate_efficiency_plot():
    plt.figure(figsize=(11, 6))
    for i in range(len(models)):
        # Bubble size represents relative Model Size in MB
        plt.scatter(params_m[i], accuracies[i], s=np.sqrt(model_sizes_mb[i])*100, 
                    color=colors[i], label=models[i], edgecolors='black', alpha=0.7)
        plt.text(params_m[i], accuracies[i] + 2, models[i], fontsize=10, fontweight='bold', ha='center')

    plt.xscale('log') # Use log scale to show small vs large models clearly
    plt.xlabel('Log Scale: Parameters (Millions)', fontsize=11, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('The Efficiency Frontier: Accuracy vs. Model Complexity', fontsize=13, fontweight='bold')
    plt.grid(True, which="both", linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig('efficiency_frontier.png', dpi=300)
    print("Saved: efficiency_frontier.png")

# --- GRAPH 3: RESOURCE FOOTPRINT (Size in MB) ---
def generate_size_chart():
    plt.figure(figsize=(10, 6))
    plt.barh(models[::-1], model_sizes_mb[::-1], color=colors[::-1], alpha=0.8)
    plt.xlabel('Model Storage Size (MB) - Lower is better for Offline Mobile Apps', fontweight='bold')
    plt.title('Project Resource Comparison: Storage Footprint', fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('resource_footprint.png', dpi=300)
    print("Saved: resource_footprint.png")

if __name__ == "__main__":
    generate_suitability_chart()
    generate_efficiency_plot()
    generate_size_chart()
    print("\nSUCCESS: All 3 report-ready graphs are now in your folder.")