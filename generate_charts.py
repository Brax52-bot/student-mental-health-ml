import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Create figures directory for repository images
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", font="DejaVu Sans")

print("--- 1/2: Generating GitHub Repository Visualizations ---")

# Data summary across tested models
model_data = {
    "Model": [
        "Logistic Regression",
        "Gradient Boosting",
        "Naive Bayes",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Decision Tree",
    ],
    "Accuracy": [0.857, 0.857, 0.810, 0.762, 0.619, 0.667, 0.619],
    "ROC-AUC": [0.791, 0.781, 0.714, 0.619, 0.548, 0.524, 0.524],
}

# ---------------------------------------------------------
# REPO CHART 1: Model Comparison (ROC-AUC)
# ---------------------------------------------------------
results_df = pd.DataFrame(model_data).sort_values(by="ROC-AUC", ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(results_df["Model"], results_df["ROC-AUC"], color="#2b5c8f", edgecolor="black", height=0.6)

for bar in bars:
    w = bar.get_width()
    ax.annotate(f" {w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(3, 0),
                textcoords="offset points", ha="left", va="center", fontweight="bold")

ax.set_xlim(0, 1.0)
ax.set_xlabel("ROC-AUC Score", fontweight="bold")
ax.set_title("Model Performance Ranking (ROC-AUC)", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# REPO CHART 2: Accuracy vs. ROC-AUC Dual Bar
# ---------------------------------------------------------
df_melted = pd.melt(pd.DataFrame(model_data), id_vars=["Model"], value_vars=["Accuracy", "ROC-AUC"],
                    var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 5))
chart = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette=["#3b82f6", "#10b981"])
plt.xticks(rotation=20, ha="right", fontweight="bold")
plt.ylim(0, 1.0)
plt.title("Supervised Learning: Accuracy vs. ROC-AUC", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/dual_metric_comparison.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# REPO CHART 3: Target Class Balance
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
target_counts = pd.Series({"No Depression": 66, "Depression": 35})
ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette=["#64748b", "#ef4444"], edgecolor="black")
plt.title("Depression Target Class Balance (N=101)", fontweight="bold")
plt.ylabel("Student Count", fontweight="bold")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontweight="bold", xytext=(0, 3), textcoords="offset points")
plt.tight_layout()
plt.savefig("figures/target_distribution.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# REPO CHART 4: Confusion Matrix (Top Model)
# ---------------------------------------------------------
cm = np.array([[13, 1], [2, 5]])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No (0)", "Yes (1)"], yticklabels=["No (0)", "Yes (1)"],
            annot_kws={"size": 14, "weight": "bold"})
plt.title("Confusion Matrix (Logistic Regression)", fontweight="bold")
plt.xlabel("Predicted Class", fontweight="bold")
plt.ylabel("Actual Class", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.close()

print("✔ Saved 4 repository figures in 'figures/'")

print("--- 2/2: Generating Google Slides Poster Visualizations ---")

# ---------------------------------------------------------
# POSTER CHART 1: Model Comparison for Poster Slide
# ---------------------------------------------------------
models = ["Logistic Reg.", "Gradient Boost.", "Naive Bayes", "Random Forest", "SVM", "KNN", "Decision Tree"]
auc_scores = [0.791, 0.781, 0.714, 0.619, 0.548, 0.524, 0.524]
accuracies = [85.7, 85.7, 81.0, 76.2, 61.9, 66.7, 61.9]

y_pos = np.arange(len(models))
fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')

bars = ax.barh(y_pos, auc_scores, color='#1e3a8a', edgecolor='black', height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('ROC-AUC Score', fontsize=13, fontweight='bold')
ax.set_title('Supervised Model Performance Ranking', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 1.0)

for bar, acc in zip(bars, accuracies):
    w = bar.get_width()
    ax.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"AUC: {w:.3f} | Acc: {acc:.1f}%",
            va='center', fontsize=11, fontweight='bold', color='#0f172a')

plt.tight_layout()
plt.savefig("poster_model_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# POSTER CHART 2: Key Mental Health Findings (Prevalence & Care Gap)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='white')

# Subplot 1: Prevalence
categories = ['Overall Mental\nHealth Issue', 'Depression\nStatus']
percentages = [63.4, 34.7]
bars1 = ax1.bar(categories, percentages, color=['#3b82f6', '#ef4444'], edgecolor='black', width=0.5)
ax1.set_ylabel('Percentage of Sample (%)', fontsize=12, fontweight='bold')
ax1.set_title('Sample Mental Health Prevalence (N=101)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
for p in bars1:
    ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 3), textcoords='offset points')

# Subplot 2: Care Gap
care_status = ['Did NOT Seek\nTreatment', 'Sought\nTreatment']
care_counts = [90.6, 9.4]
bars2 = ax2.bar(care_status, care_counts, color=['#dc2626', '#16a34a'], edgecolor='black', width=0.5)
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('90.6% Treatment Care Gap', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 100)
for p in bars2:
    ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', fontsize=12, fontweight='bold', xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.savefig("poster_mental_health_findings.png", dpi=300, bbox_inches='tight')
plt.close()

print("✔ Saved 2 poster figures: 'poster_model_performance.png' & 'poster_mental_health_findings.png'")
print("\n🎉 ALL IMAGES GENERATED SUCCESSFULLY!")