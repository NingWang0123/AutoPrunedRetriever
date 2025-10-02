import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

cr_metrics_path = 'results/compressrag_medical_data_3b.json'

# load the json file

with open(cr_metrics_path, 'r', encoding='utf-8') as f:
    cr_metrics = json.load(f)


cr_metrics_3b_df = pd.DataFrame(cr_metrics)

print(cr_metrics_3b_df.head())



cr_metrics_path = 'results/compressrag_medical_data_3b_words.json'

# load the json file

with open(cr_metrics_path, 'r', encoding='utf-8') as f:
    cr_metrics = json.load(f)

cr_metrics_3b_words_df = pd.DataFrame(cr_metrics)






# Merge
df_merged = pd.merge(cr_metrics_3b_df, cr_metrics_3b_words_df, on="id", suffixes=("_3b", "_3b_words"))

# Metrics
metrics = ["total_tokens", "correctness", "context_similarity"]

# Set a global style
sns.set_theme(style="whitegrid", context="talk")  # context="talk" makes fonts larger

# Ensure output folder exists
out_dir = "results/cr_results"
os.makedirs(out_dir, exist_ok=True)

# --- 1) Boxplots (all metrics together) ---
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for i, metric in enumerate(metrics):
    plot_data = pd.DataFrame({
        "AutoPruned Retriever default": df_merged[f"{metric}_3b"],
        "AutoPruned Retriever alternating prompt": df_merged[f"{metric}_3b_words"]
    })
    sns.boxplot(data=plot_data, ax=axes[i], palette="Set2", width=0.6, fliersize=4)

    # Titles and labels
    axes[i].set_title(f"{metric.replace('_',' ').title()} Distribution", fontsize=20, weight="bold")
    axes[i].set_ylabel(metric.replace("_"," ").title(), fontsize=16, labelpad=12)
    axes[i].set_xlabel("System", fontsize=16, labelpad=10)

    # Bigger x-axis ticks
    axes[i].tick_params(axis='x', labelsize=15, width=2, length=8)
    axes[i].tick_params(axis='y', labelsize=13)

    for tick in axes[i].get_xticklabels():
        tick.set_fontsize(15)
        tick.set_weight("bold")

    axes[i].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "boxplots_comparison_qwen3b_words.pdf"), bbox_inches="tight")
plt.show()


metrics = ["total_tokens", "correctness", "context_similarity", " retrieval_latency_sec", "gen_latency_sec"]

# --- 2) Separate Mean Bar Graphs ---
for metric in metrics:
    metric_clean = metric.strip()  # remove accidental spaces

    plt.figure(figsize=(7,6))
    mean_values = {
        "System": ["default prompt", "alternating prompt"],
        "Mean": [
            df_merged[f"{metric_clean}_3b"].mean(),
            df_merged[f"{metric_clean}_3b_words"].mean()
        ]
    }
    df_means = pd.DataFrame(mean_values)
    ax = sns.barplot(data=df_means, x="System", y="Mean", palette="Set2", width=0.5)

    # Titles and labels
    plt.title(f"Mean {metric_clean.replace('_',' ').title()}", fontsize=18, weight="bold")
    plt.ylabel("Mean Value", fontsize=14)
    plt.xlabel("")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=13, weight="bold", color='black')

    plt.tight_layout()

    # Save with clean name
    fname = f"mean_{metric_clean}_qwen3b_comp.pdf"
    plt.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.show()





# python data_analysis.py