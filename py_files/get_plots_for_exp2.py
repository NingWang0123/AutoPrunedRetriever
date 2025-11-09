from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# paths
# --------------------------------------------------------------
BASE_DIR = Path("results/results_exp2")
PLOTS_DIR = Path("plots")
OUT_ACC = PLOTS_DIR / "exp2_answer_correctness_combined.pdf"

# NEW: separate outputs for different features
OUT_INPUT_TOK   = PLOTS_DIR / "exp2_avg_input_tokens_combined.pdf"
OUT_GEN_LAT     = PLOTS_DIR / "exp2_avg_gen_latency_sec_combined.pdf"
OUT_RETR_LAT    = PLOTS_DIR / "exp2_avg_retrieval_latency_sec_combined.pdf"

METHODS_EXP1 = [
    "AutoPrunedRetriever-REBEL",
    "AutoPrunedRetriever-llm",
    "V-RAG",
    "MS-GraphRAG(local)",
    "MS-GraphRAG(global)",
    "HippoRAG2",
    "LightRAG",
    "Fast-GraphRAG",
    "RAPTOR",
    "HippoRAG",
]

cmap = plt.get_cmap("tab10")
hatches = ['/', '\\', 'x', '-', '+', '.', '*', '//', '||']


def method_from_filename(name: str) -> Optional[str]:
    name = name.lower()
    if name.startswith("apr_rebel"):
        return "AutoPrunedRetriever-REBEL"
    if name.startswith("apr_llm"):
        return "AutoPrunedRetriever-llm"
    if name.startswith("hippo2"):
        return "HippoRAG2"
    if name.startswith("light"):
        return "LightRAG"
    return None


def iter_records_from_json(data: Any):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        yield item


def collect_exp2_feature_avg(
    base_dir: Path,
    feature_name: str,
    *,
    prefixes: List[str] = ("apr", "hippo2", "light"),
    skip_suffix: str = "_gs.json",
) -> Dict[str, Dict[str, float]]:
    stem_vals: Dict[str, float] = {}
    tv_vals: Dict[str, float] = {}

    for p in base_dir.glob("*.json"):
        if p.name.endswith(skip_suffix):
            continue
        if not any(p.name.lower().startswith(pref) for pref in prefixes):
            continue

        method = method_from_filename(p.name)
        if method is None:
            continue

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        vals = []
        for rec in iter_records_from_json(data):
            if feature_name in rec:
                try:
                    vals.append(float(rec[feature_name]))
                except Exception:
                    pass

        if not vals:
            continue

        avg_val = sum(vals) / len(vals)

        low = p.name.lower()
        if "_stem" in low:
            stem_vals[method] = avg_val
        elif "_tv" in low:
            tv_vals[method] = avg_val

    return {"STEM": stem_vals, "TV": tv_vals}


def load_answer_correctness(path: Path) -> Optional[float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return data["Complex Reasoning"]["average_scores"]["answer_correctness"]
    except KeyError:
        return None


def _grouped_barplot_two_x(methods: List[str], df: pd.DataFrame, outfile: Path, ylabel: str) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    LABEL_FONTSIZE  = 16
    TICK_FONTSIZE   = 14
    LEGEND_FONTSIZE = 11
    VALUE_FONTSIZE  = 10
    SPINE_WIDTH     = 1.2
    FIGSIZE         = (10.5, 5.8)
    TOP_PAD         = 1.30

    x_labels = ["STEM", "TV"]
    x = np.arange(len(x_labels))
    m = len(methods)
    total_width = 0.84
    bar_w = total_width / m
    left = x - total_width / 2

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bar_containers = []

    for i, method in enumerate(methods):
        vals = df[method].to_numpy(dtype=float)

        master_idx = METHODS_EXP1.index(method)
        color = cmap(master_idx % cmap.N)
        hatch = hatches[master_idx % len(hatches)]

        bars = ax.bar(
            left + i * bar_w,
            vals,
            width=bar_w,
            label=method,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        for patch in bars:
            patch.set_hatch(hatch)
            patch.set_alpha(0.9)
        bar_containers.append(bars)

    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x, x_labels, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    ymax = float(np.nanmax(df.to_numpy())) if df.size else 1.0
    ax.set_ylim(0, ymax * 1.20)

    for bars in bar_containers:
        for rect in bars:
            h = rect.get_height()
            # latency might be float, so show 3 decimals
            if "correctness" in ylabel:
                label_txt = f"{h:.3f}"
            else:
                label_txt = f"{h:.3f}"
            ax.annotate(
                label_txt,
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=VALUE_FONTSIZE,
            )

    ax.legend(
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, TOP_PAD),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)


def make_answer_correctness_plot() -> None:
    stem_scores: Dict[str, float] = {}
    tv_scores: Dict[str, float] = {}

    for p in BASE_DIR.glob("*_gs.json"):
        m = method_from_filename(p.name)
        if m is None:
            continue
        acc = load_answer_correctness(p)
        if acc is None:
            continue

        low = p.name.lower()
        if "_stem" in low:
            stem_scores[m] = acc
        elif "_tv" in low:
            tv_scores[m] = acc

    data = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}
    for m, v in stem_scores.items():
        data[m]["STEM"] = v
    for m, v in tv_scores.items():
        data[m]["TV"] = v

    df = pd.DataFrame(data).T
    df = df.T

    present_methods = [
        m for m in METHODS_EXP1
        if m in df.columns and not (pd.isna(df[m]["STEM"]) and pd.isna(df[m]["TV"]))
    ]
    df = df[present_methods]

    _grouped_barplot_two_x(present_methods, df, OUT_ACC, "answer_correctness")


def make_feature_plot(feature_name: str, outfile: Path, ylabel: str) -> None:
    feature_data = collect_exp2_feature_avg(BASE_DIR, feature_name)

    data = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}
    for m, v in feature_data["STEM"].items():
        data[m]["STEM"] = v
    for m, v in feature_data["TV"].items():
        data[m]["TV"] = v

    df = pd.DataFrame(data).T
    df = df.T

    present_methods = [
        m for m in METHODS_EXP1
        if m in df.columns and not (pd.isna(df[m]["STEM"]) and pd.isna(df[m]["TV"]))
    ]
    df = df[present_methods]

    _grouped_barplot_two_x(present_methods, df, outfile, ylabel)


def main() -> None:
    # correctness
    make_answer_correctness_plot()

    # input tokens example
    make_feature_plot(
        feature_name="input_tokens",
        outfile=OUT_INPUT_TOK,
        ylabel="avg input_tokens",
    )

    # your two latency features, each to its OWN file
    make_feature_plot(
        feature_name="gen_latency_sec",
        outfile=OUT_GEN_LAT,
        ylabel="avg gen_latency_sec",
    )

    make_feature_plot(
        feature_name="retrieval_latency_sec",
        outfile=OUT_RETR_LAT,
        ylabel="avg retrieval_latency_sec",
    )


if __name__ == "__main__":
    main()




# python get_plots_for_exp2.py