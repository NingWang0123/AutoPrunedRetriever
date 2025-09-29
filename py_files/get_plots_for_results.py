import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_cr_all = pd.read_csv('results/cr_results/cr_combined_results.csv')

df_cr_all['correctness_sbert'] = df_cr_all['correctness']
df_cr_all['context_similarity_sbert'] = df_cr_all['context']
df_no_idk = pd.read_csv('results/light_rag_results/lightrag_eval_results.csv')

# filter the same 'med_id'


# compare column 'total_tokens'


# compare column 'correctness_sbert'


# compare column 'context_similarity_sbert'



# Merge on med_id to align the same questions
df_merged = pd.merge(df_cr_all, df_no_idk, on="med_id", suffixes=("_apr", "_light"))

# Metrics to compare
metrics = ["total_tokens", "correctness_sbert", "context_similarity_sbert"]

# Boxplots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df_cr_all = pd.read_csv('results/cr_results/cr_combined_results.csv')
df_cr_all['correctness_sbert'] = df_cr_all['correctness']
df_cr_all['context_similarity_sbert'] = df_cr_all['context']

df_no_idk = pd.read_csv('results/light_rag_results/lightrag_eval_results.csv')

# Rename index column -> med_id if needed
if 'med_id' not in df_cr_all.columns:
    df_cr_all = df_cr_all.rename(columns={df_cr_all.columns[0]: 'med_id'})
if 'med_id' not in df_no_idk.columns:
    df_no_idk = df_no_idk.rename(columns={df_no_idk.columns[0]: 'med_id'})

# Merge
df_merged = pd.merge(df_cr_all, df_no_idk, on="med_id", suffixes=("_apr", "_light"))

# Metrics
metrics = ["total_tokens", "correctness_sbert", "context_similarity_sbert"]

# --- 1) Boxplots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, metric in enumerate(metrics):
    plot_data = pd.DataFrame({
        "AutoPruned Retriever": df_merged[f"{metric}_apr"],
        "LightRAG": df_merged[f"{metric}_light"]
    })
    sns.boxplot(data=plot_data, ax=axes[i], palette="Set2")
    axes[i].set_title(f"{metric} Distribution", fontsize=14)
    axes[i].set_ylabel(metric)
    axes[i].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# --- 2) Mean bar plots ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df_cr_all = pd.read_csv('results/cr_results/cr_combined_results.csv')
df_cr_all['correctness_sbert'] = df_cr_all['correctness']
df_cr_all['context_similarity_sbert'] = df_cr_all['context']

df_no_idk = pd.read_csv('results/light_rag_results/lightrag_eval_results.csv')

# Rename index column -> med_id if needed
if 'med_id' not in df_cr_all.columns:
    df_cr_all = df_cr_all.rename(columns={df_cr_all.columns[0]: 'med_id'})
if 'med_id' not in df_no_idk.columns:
    df_no_idk = df_no_idk.rename(columns={df_no_idk.columns[0]: 'med_id'})

# Merge
df_merged = pd.merge(df_cr_all, df_no_idk, on="med_id", suffixes=("_apr", "_light"))

# Metrics
metrics = ["total_tokens", "correctness_sbert", "context_similarity_sbert"]

# --- 1) Boxplots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, metric in enumerate(metrics):
    plot_data = pd.DataFrame({
        "AutoPruned Retriever": df_merged[f"{metric}_apr"],
        "LightRAG": df_merged[f"{metric}_light"]
    })
    sns.boxplot(data=plot_data, ax=axes[i], palette="Set2")
    axes[i].set_title(f"{metric} Distribution", fontsize=14)
    axes[i].set_ylabel(metric)
    axes[i].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# --- 2) Separate Mean Bar Graphs ---
for metric in metrics:
    plt.figure(figsize=(6,5))
    mean_values = {
        "System": ["AutoPruned Retriever", "LightRAG"],
        "Mean": [
            df_merged[f"{metric}_apr"].mean(),
            df_merged[f"{metric}_light"].mean()
        ]
    }
    df_means = pd.DataFrame(mean_values)
    ax = sns.barplot(data=df_means, x="System", y="Mean", palette="Set2")
    plt.title(f"Mean {metric}", fontsize=16)
    plt.ylabel("Mean Value")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()




# python get_plots_for_results.py