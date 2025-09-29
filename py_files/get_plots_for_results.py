import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


df_cr_all = pd.read_csv('results/cr_results/cr_combined_results.csv')

df_cr_all['correctness_sbert'] = df_cr_all['correctness']
df_cr_all['context_similarity_sbert'] = df_cr_all['context']
df_no_idk = pd.read_csv('results/light_rag_results/lightrag_eval_results.csv')

# filter the same 'med_id'


# compare column 'total_tokens'


# compare column 'correctness_sbert'


# compare column 'context_similarity_sbert'



# Merge on med_id to align the same questions


# Merge
df_merged = pd.merge(df_cr_all, df_no_idk, on="med_id", suffixes=("_apr", "_light"))

# Metrics
metrics = ["total_tokens", "correctness_sbert", "context_similarity_sbert"]

# Set a global style
sns.set_theme(style="whitegrid", context="talk")  # context="talk" makes fonts larger

# Ensure output folder exists
out_dir = "results/cr_results"
os.makedirs(out_dir, exist_ok=True)

# --- 1) Boxplots (all metrics together) ---
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for i, metric in enumerate(metrics):
    plot_data = pd.DataFrame({
        "AutoPruned Retriever": df_merged[f"{metric}_apr"],
        "LightRAG": df_merged[f"{metric}_light"]
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
plt.savefig(os.path.join(out_dir, "boxplots_comparison_qwen3b.pdf"), bbox_inches="tight")
plt.show()

# --- 2) Separate Mean Bar Graphs ---
for metric in metrics:
    plt.figure(figsize=(7,6))
    mean_values = {
        "System": ["AutoPruned Retriever", "LightRAG"],
        "Mean": [
            df_merged[f"{metric}_apr"].mean(),
            df_merged[f"{metric}_light"].mean()
        ]
    }
    df_means = pd.DataFrame(mean_values)
    ax = sns.barplot(data=df_means, x="System", y="Mean", palette="Set2", width=0.5)

    # Titles and labels
    plt.title(f"Mean {metric.replace('_',' ').title()}", fontsize=18, weight="bold")
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

    # Save with good name
    fname = f"mean_{metric}_qwen3b.pdf"
    plt.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    plt.show()





# python get_plots_for_results.py