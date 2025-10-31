from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Config ---------------------------
RESULTS_NOVEL   = Path("results/compressrag_novel_data_openai_test_novel_v4.json")
RESULTS_MEDICAL = Path("results/compressrag_medical_data_openai_test_new_v3_for_aprv4.json")

# Fallbacks (optional)
FALLBACK_NOVEL   = Path("/mnt/data/compressrag_novel_data_openai_test_novel_v4.json")
FALLBACK_MEDICAL = Path("/mnt/data/compressrag_medical_data_openai_test_new_v3_for_aprv4.json")

# Output
OUT_COMBINED = Path("plots/avg_tokens_combined.pdf")
# --------------------------------------------------------------

# Visuals
cmap = plt.get_cmap("tab10")
hatches = ['/', '\\', 'x', '-', '+', '.', '*', '//', '||']

def _resolve(p: Path, fallback: Path) -> Path:
    if p.exists():
        return p
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find file at {p} or fallback {fallback}")

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _extract_tokens(data: Any) -> List[float]:
    """
    Accepts either:
    - list[dict{..."input_tokens": <num>...}]
    - dict[str -> list[dict{..."input_tokens": <num>...}]]
    Returns list of floats.
    """
    out: List[float] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "input_tokens" in item:
                try:
                    out.append(float(item["input_tokens"]))
                except Exception:
                    pass
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "input_tokens" in item:
                        try:
                            out.append(float(item["input_tokens"]))
                        except Exception:
                            pass
    return out

def _avg(nums: List[float]) -> float:
    return (sum(nums) / len(nums)) if nums else float("nan")

def _format_int(n: float | int) -> str:
    try:
        return f"{int(round(n)):,}"
    except Exception:
        return "NaN"

def _grouped_barplot_two_x(
    methods: List[str],
    df: pd.DataFrame,         # index: ["Novel","Medical"], columns: methods
    title: str,               # ignored (kept for signature compatibility)
    outfile: Path
) -> None:
    """Grouped bars with x ∈ {Novel, Medical}; shared legend on top, no title, value labels on bars."""
    import numpy as np
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # ---- sizes you can tune ----
    LABEL_FONTSIZE  = 16
    TICK_FONTSIZE   = 14
    LEGEND_FONTSIZE = 11
    VALUE_FONTSIZE  = 12
    SPINE_WIDTH     = 1.2
    FIGSIZE         = (10.5, 5.8)
    TOP_PAD         = 1.22  # space for legend + value labels
    # ----------------------------

    df = df.loc[["Novel", "Medical"], methods].astype(float)
    x_labels = ["Novel", "Medical"]
    x = np.arange(len(x_labels))  # [0, 1]
    m = len(methods)
    total_width = 0.84
    bar_w = total_width / m
    left = x - total_width / 2

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Draw bars and keep handles for labeling
    bar_containers = []
    for i, method in enumerate(methods):
        vals = df[method].to_numpy(dtype=float)  # [Novel, Medical]
        color = cmap(i % cmap.N)
        hatch = hatches[i % len(hatches)]
        bars = ax.bar(left + i * bar_w, vals, width=bar_w, label=method, color=color)
        for patch in bars:
            patch.set_hatch(hatch)
        bar_containers.append(bars)

    # Axes cosmetics
    ax.set_ylabel("Avg Tokens", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x, x_labels, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    # Room for labels above bars: bump the y-limit a bit
    ymax = float(np.nanmax(df.to_numpy())) if df.size else 1.0
    ax.set_ylim(0, ymax * 1.12)

    # Value labels on top of each bar
    for bars in bar_containers:
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                _format_int(h),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # Shared legend at the top
    ax.legend(
        ncol=3, fontsize=LEGEND_FONTSIZE, loc="upper center",
        bbox_to_anchor=(0.5, TOP_PAD), frameon=False
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    # ---- load + compute AutoPrunedRetriever averages from JSON ----
    p_novel = _resolve(RESULTS_NOVEL, FALLBACK_NOVEL)
    p_med   = _resolve(RESULTS_MEDICAL, FALLBACK_MEDICAL)

    novel_avg   = _avg(_extract_tokens(_load_json(p_novel)))
    medical_avg = _avg(_extract_tokens(_load_json(p_med)))
    auto_novel   = int(round(novel_avg))   if not math.isnan(novel_avg)   else None
    auto_medical = int(round(medical_avg)) if not math.isnan(medical_avg) else None

    # ---- baselines (fill in with your real numbers as needed) ----
    vrag       = {"Novel": 879,     "Medical": 954}
    ms_local   = {"Novel": 38707,   "Medical": 39821}
    ms_global  = {"Novel": 331375,  "Medical": 332881}
    hippo2     = {"Novel": 1008,    "Medical": 1020}
    lightrag   = {"Novel": 100832,  "Medical": 100310}
    fast_graph = {"Novel": 4204,    "Medical": 4298}
    raptor     = {"Novel": 3441,    "Medical": 3510}
    hippo      = {"Novel": 7208,    "Medical": 7342}

    methods = [
        "AutoPrunedRetriever",
        "V-RAG",
        "MS-GraphRAG(local)",
        "MS-GraphRAG(global)",
        "HippoRAG2",
        "LightRAG",
        "Fast-GraphRAG",
        "RAPTOR",
        "HippoRAG",
    ]

    data: Dict[str, Dict[str, float | None]] = {
        "AutoPrunedRetriever": {"Novel": auto_novel, "Medical": auto_medical},
        "V-RAG": vrag,
        "MS-GraphRAG(local)": ms_local,
        "MS-GraphRAG(global)": ms_global,
        "HippoRAG2": hippo2,
        "LightRAG": lightrag,
        "Fast-GraphRAG": fast_graph,
        "RAPTOR": raptor,
        "HippoRAG": hippo,
    }

    # ---- dataframe in desired order ----
    df = pd.DataFrame(data)
    df = df.loc[["Novel", "Medical"], methods]

    # ---- plot single combined figure ----
    _grouped_barplot_two_x(methods, df, "ignored", OUT_COMBINED)

    # ---- simple console summary ----
    print("AutoPrunedRetriever rounded averages from JSON:")
    print(f"  Novel  : {auto_novel}")
    print(f"  Medical: {auto_medical}")
    print(f"Saved plot: {OUT_COMBINED}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())




# python prompts_tokens_graphrag.py