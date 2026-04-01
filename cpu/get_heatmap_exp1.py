import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
heatmap_path = PLOTS_DIR / "heatmap.pdf"

# ------------------ data ------------------
novel_rows = [
    ("RAG (w/o rerank)",              58.76, 37.35, 41.35, 15.12, 50.08, 82.53, 41.52, 47.46, 37.84),
    ("RAG (w/ rerank)",               60.92, 36.08, 42.93, 15.39, 51.30, 83.64, 38.26, 49.21, 40.04),
    ("MS-GraphRAG (local)",           49.29, 26.11, 50.93, 24.09, 64.40, 75.58, 39.10, 55.44, 35.65),
    ("HippoRAG",                      52.93, 26.65, 38.52, 11.16, 48.70, 85.55, 38.85, 71.53, 38.97),
    ("HippoRAG2",                     60.14, 31.35, 53.38, 33.42, 64.10, 70.84, 48.28, 49.84, 30.95),
    ("LightRAG",                      58.62, 35.72, 49.07, 24.16, 48.85, 63.05, 23.80, 57.28, 25.01),
    ("Fast-GraphRAG",                 56.95, 35.90, 48.55, 21.12, 56.41, 80.82, 46.18, 57.19, 36.99),
    ("RAPTOR",                        49.25, 23.74, 38.59, 11.66, 47.10, 82.33, 38.01, 70.85, 35.88),
    ("Lazy-GraphRAG",                 51.65, 36.97, 49.22, 23.48, 58.29, 76.94, 43.23, 50.69, 39.74),
    ("AutoPrunedRetriever-REBEL",     49.25, 38.02, 63.02, 31.25, 82.55, 83.95, 59.94, 25.78, 21.21),
    ("AutoPrunedRetriever-llm",       45.99, 26.99, 62.80, 35.35, 83.10, 83.86, 62.97, 34.40, 22.13),
]

medical_rows = [
    ("RAG (w/o rerank)",              63.72, 29.21, 57.61, 13.98, 63.72, 77.34, 58.94, 35.88, 57.87),
    ("RAG (w/ rerank)",               64.73, 30.75, 58.64, 15.57, 65.75, 78.54, 60.61, 36.74, 58.72),
    ("MS-GraphRAG (local)",           38.63, 26.80, 47.04, 21.99, 41.87, 22.98, 53.11, 32.65, 39.42),
    ("HippoRAG",                      56.14, 20.95, 55.87, 13.57, 59.86, 62.73, 64.43, 69.21, 65.56),
    ("HippoRAG2",                     66.28, 36.69, 61.98, 36.97, 63.08, 46.13, 68.05, 58.78, 51.54),
    ("LightRAG",                      63.32, 37.19, 61.32, 24.98, 63.14, 51.16, np.nan, np.nan, np.nan),
    ("Fast-GraphRAG",                 60.93, 31.04, 61.73, 21.37, 67.88, 52.07, 65.93, 56.07, 44.73),
    ("RAPTOR",                        54.07, 17.93, 53.20, 11.73, 58.73, 78.28, np.nan, np.nan, np.nan),
    ("Lazy-GraphRAG",                 60.25, 31.66, 47.82, 22.68, 57.28, 55.92, 62.22, 30.95, 43.79),
    ("AutoPrunedRetriever-REBEL",     61.28, 32.96, 72.49, 30.79, 68.78, 40.15, 64.04, 32.19, 11.12),
    ("AutoPrunedRetriever-llm",       61.25, 34.69, 71.59, 31.11, 70.14, 40.59, 65.02, 33.06, 28.62),
]

cols = [
    "FR_ACC", "FR_ROUGE",
    "CR_ACC", "CR_ROUGE",
    "CS_ACC", "CS_Cov",
    "CG_ACC", "CG_Cov", "CG_FS_Cov",
]

df_novel = pd.DataFrame(novel_rows, columns=["Model"] + cols).set_index("Model")
df_med   = pd.DataFrame(medical_rows, columns=["Model"] + cols).set_index("Model")

# ------------------ plot ------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(19, 8),
    gridspec_kw={"width_ratios": [1, 1.05]}
)

cmap = sns.cm.rocket_r

# left heatmap with explicit yticklabels
sns.heatmap(
    df_novel,
    ax=axes[0],
    cmap=cmap,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.5,
    linecolor="white",
    cbar=False,
    vmin=0,
    vmax=100,
    yticklabels=df_novel.index.tolist(),  # <--- force model names here
)
axes[0].set_title("Novel Dataset", fontsize=12)
axes[0].tick_params(axis="x", rotation=45)
axes[0].tick_params(axis="y", labelsize=8)

# make a dedicated colorbar axis on the right side of the figure
# [left, bottom, width, height] in figure fraction
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])

# right heatmap, share y positions but hide labels
sns.heatmap(
    df_med,
    ax=axes[1],
    cmap=cmap,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    cbar_ax=cax,
    vmin=0,
    vmax=100,
    yticklabels=False,  # <--- we only show on the left
)
axes[1].set_title("Medical Dataset", fontsize=12)
axes[1].tick_params(axis="x", rotation=45)

# give LOTS of room for the left labels
plt.subplots_adjust(left=0.35, right=0.9, wspace=0.08)

# IMPORTANT: don't use bbox_inches="tight"
plt.savefig(heatmap_path, dpi=1200, bbox_inches="tight")
plt.show()






# python get_heatmap_exp1.py