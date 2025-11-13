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

# single-output files
OUT_ACC        = PLOTS_DIR / "exp2_answer_correctness_combined.pdf"
OUT_INPUT_TOK  = PLOTS_DIR / "exp2_avg_input_tokens_combined.pdf"
OUT_GEN_LAT    = PLOTS_DIR / "exp2_avg_gen_latency_sec_combined.pdf"
OUT_RETR_LAT   = PLOTS_DIR / "exp2_avg_retrieval_latency_sec_combined.pdf"

# pair-output files
OUT_LAT_PAIR   = PLOTS_DIR / "exp2_latency_pair.pdf"
OUT_GS_PAIR    = PLOTS_DIR / "exp2_gs_pair.pdf"       # <-- new: correctness + rouge side-by-side
OUT_GS_FULL   = PLOTS_DIR / "exp2_gs_full.pdf"       # <-- new: correctness + rouge side-by-side
OUT_ROUGE      = PLOTS_DIR / "exp2_rouge_score_combined.pdf"  # optional single rouge plot
OUT_INPUT_TOK   = PLOTS_DIR / "exp2_avg_input_tokens_combined.pdf"
OUT_GEN_LAT     = PLOTS_DIR / "exp2_avg_gen_latency_sec_combined.pdf"
OUT_RETR_LAT    = PLOTS_DIR / "exp2_avg_retrieval_latency_sec_combined.pdf"
OUT_LAT_PAIR    = PLOTS_DIR / "exp2_latency_pair.pdf"   # <--- new: 2 plots side by side
OUT_TOK_PAIR    = PLOTS_DIR / "exp2_tokens_pair.pdf"   # <--- new: 2 plots side by side

# master order from exp1 (defines color + hatch slots)
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


# --------------------------------------------------------------
# helpers
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# collect feature from non-_gs (input_tokens, latencies, etc.)
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# collect from *_gs.json : BOTH correctness and rouge
# --------------------------------------------------------------
def collect_exp2_gs_metrics(base_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    returns:
    {
      "STEM": {
         method: {
            "answer_correctness": ...,
            "rouge_score": ...
         },
         ...
      },
      "TV": { ... }
    }
    """
    stem: Dict[str, Dict[str, float]] = {}
    tv: Dict[str, Dict[str, float]] = {}

    for p in base_dir.glob("*_gs.json"):
        method = method_from_filename(p.name)
        if method is None:
            continue

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # try to read both metrics
        ac = None
        rouge = None
        try:
            ac = data["Complex Reasoning"]["average_scores"]["answer_correctness"]
        except KeyError:
            pass
        try:
            rouge = data["Complex Reasoning"]["average_scores"]["rouge_score"]
        except KeyError:
            pass

        metrics_for_method: Dict[str, float] = {}
        if ac is not None:
            metrics_for_method["answer_correctness"] = float(ac)
        if rouge is not None:
            metrics_for_method["rouge_score"] = float(rouge)

        if not metrics_for_method:
            continue

        low = p.name.lower()
        if "_stem" in low:
            stem.setdefault(method, {}).update(metrics_for_method)
        elif "_tv" in low:
            tv.setdefault(method, {}).update(metrics_for_method)

    return {"STEM": stem, "TV": tv}


# --------------------------------------------------------------
# shared plotting core
# --------------------------------------------------------------
def _grouped_barplot_two_x(methods: List[str], df: pd.DataFrame, ax: plt.Axes, ylabel: str):
    LABEL_FONTSIZE  = 20
    TICK_FONTSIZE   = 20
    VALUE_FONTSIZE  = 9
    SPINE_WIDTH     = 1.2

    x_labels = ["STEM", "TV"]
    x = np.arange(len(x_labels))
    m = len(methods)
    total_width = 0.84
    bar_w = total_width / m
    left = x - total_width / 2

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
            # both correctness and rouge are floats
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


# --------------------------------------------------------------
# single plots
# --------------------------------------------------------------
def make_feature_plot(feature_name: str, outfile: Path, ylabel: str) -> None:
    feature_data = collect_exp2_feature_avg(BASE_DIR, feature_name)

    # base df
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

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    _grouped_barplot_two_x(present_methods, df, ax, ylabel)
    ax.legend(
        ncol=3,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------
# GS pair: correctness + rouge side-by-side, shared legend
# --------------------------------------------------------------
def make_gs_plot_pair(outfile: Path) -> None:
    gs = collect_exp2_gs_metrics(BASE_DIR)

    # build dfs for correctness and rouge
    # start from master
    base_corr = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}
    base_rouge = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}

    for m, metrics in gs["STEM"].items():
        if "answer_correctness" in metrics:
            base_corr[m]["STEM"] = metrics["answer_correctness"]
        if "rouge_score" in metrics:
            base_rouge[m]["STEM"] = metrics["rouge_score"]

    for m, metrics in gs["TV"].items():
        if "answer_correctness" in metrics:
            base_corr[m]["TV"] = metrics["answer_correctness"]
        if "rouge_score" in metrics:
            base_rouge[m]["TV"] = metrics["rouge_score"]

    df_corr = pd.DataFrame(base_corr).T
    df_corr = df_corr.T
    df_rouge = pd.DataFrame(base_rouge).T
    df_rouge = df_rouge.T

    # methods present in either metric
    present_methods = []
    for m in METHODS_EXP1:
        corr_has = m in df_corr.columns and not (pd.isna(df_corr[m]["STEM"]) and pd.isna(df_corr[m]["TV"]))
        rouge_has = m in df_rouge.columns and not (pd.isna(df_rouge[m]["STEM"]) and pd.isna(df_rouge[m]["TV"]))
        if corr_has or rouge_has:
            present_methods.append(m)

    df_corr = df_corr[present_methods]
    df_rouge = df_rouge[present_methods]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8),gridspec_kw={"wspace": 0.3})

    _grouped_barplot_two_x(present_methods, df_corr, axes[0], "Answer Correctness")
    # axes[0].set_title("answer_correctness", fontsize=12)

    _grouped_barplot_two_x(present_methods, df_rouge, axes[1], "Rouge Score")
    # axes[1].set_title("rouge_score", fontsize=12)

    # shared legend
    handles = []
    labels = []
    for m in present_methods:
        master_idx = METHODS_EXP1.index(m)
        color = cmap(master_idx % cmap.N)
        hatch = hatches[master_idx % len(hatches)]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.6, hatch=hatch)
        handles.append(patch)
        labels.append(m)

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------
# latency pair (kept from before)
# --------------------------------------------------------------
def make_feature_plot_pair(
    feature_name_left: str,
    feature_name_right: str,
    outfile: Path,
    ylabel_left: str,
    ylabel_right: str,
) -> None:
    left_data = collect_exp2_feature_avg(BASE_DIR, feature_name_left)
    right_data = collect_exp2_feature_avg(BASE_DIR, feature_name_right)

    base_left = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}
    for m, v in left_data["STEM"].items():
        base_left[m]["STEM"] = v
    for m, v in left_data["TV"].items():
        base_left[m]["TV"] = v
    df_left = pd.DataFrame(base_left).T
    df_left = df_left.T

    base_right = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}
    for m, v in right_data["STEM"].items():
        base_right[m]["STEM"] = v
    for m, v in right_data["TV"].items():
        base_right[m]["TV"] = v
    df_right = pd.DataFrame(base_right).T
    df_right = df_right.T

    present_methods = []
    for m in METHODS_EXP1:
        left_has = m in df_left.columns and not (pd.isna(df_left[m]["STEM"]) and pd.isna(df_left[m]["TV"]))
        right_has = m in df_right.columns and not (pd.isna(df_right[m]["STEM"]) and pd.isna(df_right[m]["TV"]))
        if left_has or right_has:
            present_methods.append(m)

    df_left = df_left[present_methods]
    df_right = df_right[present_methods]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8),gridspec_kw={"wspace": 0.3})

    # fig.subplots_adjust(wspace=2)

    _grouped_barplot_two_x(present_methods, df_left, axes[0], ylabel_left)
    # axes[0].set_title(ylabel_left, fontsize=12)

    _grouped_barplot_two_x(present_methods, df_right, axes[1], ylabel_right)
    # axes[1].set_title(ylabel_right, fontsize=12)

    # shared legend
    handles = []
    labels = []
    for m in present_methods:
        master_idx = METHODS_EXP1.index(m)
        color = cmap(master_idx % cmap.N)
        hatch = hatches[master_idx % len(hatches)]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.6, hatch=hatch)
        handles.append(patch)
        labels.append(m)

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)

# add near your other OUT_* paths
OUT_GRAPH_3_PANEL = PLOTS_DIR / "exp2_graph_tokens_and_workspace_3panel.pdf"


def _grouped_barplot_two_x_int(methods, df, ax, ylabel):
    """
    Same as _grouped_barplot_two_x but labels as integers.
    """
    x_labels = ["STEM", "TV"]
    x = np.arange(len(x_labels))
    m = len(methods)
    total_width = 0.84
    bar_w = total_width / m
    left = x - total_width / 2

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

    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xticks(x, x_labels, fontsize=20)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    ymax = float(np.nanmax(df.to_numpy())) if df.size else 1.0
    ax.set_ylim(0, ymax * 1.20)

    # integer labels
    for bars in bar_containers:
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{int(round(h)):,}",   # <-- integer with commas
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def make_graph_tokens_workspace_3panel(outfile: Path) -> None:
    methods = [
        "AutoPrunedRetriever-REBEL",
        "AutoPrunedRetriever-llm",
        "HippoRAG2",
        "LightRAG"
    ]

    prompt_data = {
        "AutoPrunedRetriever-REBEL": {"STEM": 0,       "TV": 0},
        "AutoPrunedRetriever-llm":   {"STEM": 1204757, "TV": 2835689},
        "HippoRAG2":                 {"STEM": 2212066, "TV": 5016941},
        "LightRAG":                  {"STEM": 4822577, "TV": 11161008}
    }
    completion_data = {
        "AutoPrunedRetriever-REBEL": {"STEM": 0,      "TV": 0},
        "AutoPrunedRetriever-llm":   {"STEM": 761816, "TV": 2494504},
        "HippoRAG2":                 {"STEM": 624078, "TV": 1957095},
        "LightRAG":                  {"STEM": 764044, "TV": 1980769}
    }
    ws_data = {
        "AutoPrunedRetriever-REBEL": {"STEM": 68,   "TV": 542},
        "AutoPrunedRetriever-llm":   {"STEM": 705,  "TV": 2292},
        "HippoRAG2":                 {"STEM": 259,  "TV": 443},
        "LightRAG":                  {"STEM": 154, "TV": 370}
    }

    import pandas as pd
    import matplotlib.pyplot as plt

    df_prompt = pd.DataFrame({m: prompt_data[m] for m in methods}).T
    df_prompt = df_prompt.T
    df_comp = pd.DataFrame({m: completion_data[m] for m in methods}).T
    df_comp = df_comp.T
    df_ws = pd.DataFrame({m: ws_data[m] for m in methods}).T
    df_ws = df_ws.T

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharey=False)

    _grouped_barplot_two_x_int(methods, df_prompt, axes[0], "Graph Prompt Tokens")
    # axes[0].set_title("Graph prompt tokens", fontsize=12)

    _grouped_barplot_two_x_int(methods, df_comp, axes[1], "Graph Completion Tokens")
    # axes[1].set_title("Graph completion tokens", fontsize=12)

    _grouped_barplot_two_x_int(methods, df_ws, axes[2], "Workspace Size (MB)")
    # axes[2].set_title("Workspace size (MB)", fontsize=12)

    # shared legend
    handles, labels = [], []
    for m in methods:
        master_idx = METHODS_EXP1.index(m)
        color = cmap(master_idx % cmap.N)
        hatch = hatches[master_idx % len(hatches)]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.6, hatch=hatch)
        handles.append(patch)
        labels.append(m)

    fig.legend(
        handles,
        labels,
        ncol=3,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)



# add near other OUT_* paths
OUT_APR_CODEBOOK_DIFF = PLOTS_DIR / "exp2_apr_codebook_size_diff.pdf"



# add/replace this in your file
def make_apr_codebook_diff_from_first(outfile: Path) -> None:
    """
    APR only, non-gs.
    For each file, diff is defined as:
        diff[i] = meta_codebook_json_MB[i] - meta_codebook_json_MB[0]
    so everything is measured against the first question of that run.

    Left: STEM (rebel + llm)
    Right: TV  (rebel + llm)
    """
    base_dir = BASE_DIR

    def _load_vals(path: Path) -> list[float]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        vals: list[float] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "meta_codebook_json_MB" in item:
                    try:
                        vals.append(float(item["meta_codebook_json_MB"]))
                    except Exception:
                        pass
        elif isinstance(data, dict) and "meta_codebook_json_MB" in data:
            vals.append(float(data["meta_codebook_json_MB"]))
        return vals

    # load 4 apr files
    rebel_stem_vals = _load_vals(base_dir / "apr_rebel_stem.json")
    rebel_tv_vals   = _load_vals(base_dir / "apr_rebel_tv.json")
    llm_stem_vals   = _load_vals(base_dir / "apr_llm_stem.json")
    llm_tv_vals     = _load_vals(base_dir / "apr_llm_tv.json")

    def _diff_from_first(vals: list[float]) -> list[float]:
        if not vals:
            return []
        base = vals[0]
        return [v - base for v in vals]

    rebel_stem_diff = _diff_from_first(rebel_stem_vals)
    rebel_tv_diff   = _diff_from_first(rebel_tv_vals)
    llm_stem_diff   = _diff_from_first(llm_stem_vals)
    llm_tv_diff     = _diff_from_first(llm_tv_vals)

    # simple line smoothing via interpolation (keeps shape but looks nicer)
    import numpy as np
    import matplotlib.pyplot as plt

    def _smooth_xy(x, y, num=200):
        if len(x) < 2:
            return x, y
        x_new = np.linspace(x[0], x[-1], num)
        y_new = np.interp(x_new, x, y)
        return x_new, y_new

    # fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8),gridspec_kw={"wspace": 0.3})

    # ----- STEM -----
    ax = axes[0]
    ax.axhline(0, color="gray", linewidth=0.6)
    # rebel
    if rebel_stem_diff:
        x = np.arange(1, len(rebel_stem_diff) + 1)
        midx = METHODS_EXP1.index("AutoPrunedRetriever-REBEL")
        color = cmap(midx % cmap.N)
        # raw points
        ax.plot(x, rebel_stem_diff, "o", color=color, alpha=0.35)
        # smooth
        xs, ys = _smooth_xy(x, np.array(rebel_stem_diff))
        ax.plot(xs, ys, "-", color=color, label="AutoPrunedRetriever-REBEL")
    # llm
    if llm_stem_diff:
        x = np.arange(1, len(llm_stem_diff) + 1)
        midx = METHODS_EXP1.index("AutoPrunedRetriever-llm")
        color = cmap(midx % cmap.N)
        ax.plot(x, llm_stem_diff, "o", color=color, alpha=0.35)
        xs, ys = _smooth_xy(x, np.array(llm_stem_diff))
        ax.plot(xs, ys, "-", color=color, label="AutoPrunedRetriever-llm")
    # ax.set_title("APR codebook Δ from first (STEM)")
    ax.set_xlabel("Question Index",fontsize = 20)
    ax.set_ylabel("Δ MB from first",fontsize = 20)

    # ----- TV -----
    ax = axes[1]
    ax.axhline(0, color="gray", linewidth=0.6)
    if rebel_tv_diff:
        x = np.arange(1, len(rebel_tv_diff) + 1)
        midx = METHODS_EXP1.index("AutoPrunedRetriever-REBEL")
        color = cmap(midx % cmap.N)
        ax.plot(x, rebel_tv_diff, "o", color=color, alpha=0.35)
        xs, ys = _smooth_xy(x, np.array(rebel_tv_diff))
        ax.plot(xs, ys, "-", color=color, label="AutoPrunedRetriever-REBEL")
    if llm_tv_diff:
        x = np.arange(1, len(llm_tv_diff) + 1)
        midx = METHODS_EXP1.index("AutoPrunedRetriever-llm")
        color = cmap(midx % cmap.N)
        ax.plot(x, llm_tv_diff, "o", color=color, alpha=0.35)
        xs, ys = _smooth_xy(x, np.array(llm_tv_diff))
        ax.plot(xs, ys, "-", color=color, label="AutoPrunedRetriever-llm")
    # ax.set_title("APR codebook Δ from first (TV)")
    ax.set_xlabel("Question Index",fontsize = 20)
    ax.set_ylabel("Δ MB from first",fontsize = 20)

    # shared legend, same style as bars
    handles, labels = [], []
    for method_name in ["AutoPrunedRetriever-REBEL", "AutoPrunedRetriever-llm"]:
        # include only if we found data
        has_data = (
            (method_name == "AutoPrunedRetriever-REBEL" and (rebel_stem_diff or rebel_tv_diff)) or
            (method_name == "AutoPrunedRetriever-llm" and (llm_stem_diff or llm_tv_diff))
        )
        if not has_data:
            continue
        midx = METHODS_EXP1.index(method_name)
        color = cmap(midx % cmap.N)
        hatch = hatches[midx % len(hatches)]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black",
                              linewidth=0.6, hatch=hatch)
        handles.append(patch)
        labels.append(method_name)

    fig.legend(
        handles,
        labels,
        ncol=2,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)



from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_gs_plot_full(outfile: Path) -> None:
    """
    Plot answer correctness and ROUGE-L for exactly these four methods:
    - HippoRAG2
    - LightRAG
    - AutoPrunedRetriever-REBEL
    - AutoPrunedRetriever-llm
    across four datasets: Medical-CR, Novel-CR (hardcoded), STEM, TV.
    """
    METHODS_FOUR = [
        "AutoPrunedRetriever-REBEL",
        "AutoPrunedRetriever-llm",
        "HippoRAG2",
        "LightRAG",
    ]

    # 1) hardcode Medical / Novel CR from the table, but with the SHORT names
    medical_cr = {
        "HippoRAG2":               {"answer_correctness": 61.98/100, "rouge_score": 36.97/100},
        "LightRAG":                {"answer_correctness": 61.32/100, "rouge_score": 24.98/100},
        "AutoPrunedRetriever-REBEL": {"answer_correctness": 72.49/100, "rouge_score": 30.79/100},
        "AutoPrunedRetriever-llm":   {"answer_correctness": 71.59/100, "rouge_score": 31.11/100},
    }

    novel_cr = {
        "HippoRAG2":               {"answer_correctness": 53.38/100, "rouge_score": 33.42/100},
        "LightRAG":                {"answer_correctness": 49.07/100, "rouge_score": 24.16/100},
        "AutoPrunedRetriever-REBEL": {"answer_correctness": 63.02/100, "rouge_score": 31.25/100},
        "AutoPrunedRetriever-llm":   {"answer_correctness": 62.80/100, "rouge_score": 35.35/100},
    }

    # 2) get STEM / TV from your existing logs
    gs = collect_exp2_gs_metrics(BASE_DIR)

    DATASETS = ["Medical", "Novel", "STEM", "TV"]
    base_corr = {m: {ds: float("nan") for ds in DATASETS} for m in METHODS_FOUR}
    base_rouge = {m: {ds: float("nan") for ds in DATASETS} for m in METHODS_FOUR}

    # fill Medical / Novel
    for m in METHODS_FOUR:
        base_corr[m]["Medical"] = medical_cr[m]["answer_correctness"]
        base_rouge[m]["Medical"] = medical_cr[m]["rouge_score"]
        base_corr[m]["Novel"] = novel_cr[m]["answer_correctness"]
        base_rouge[m]["Novel"] = novel_cr[m]["rouge_score"]

    # fill STEM
    if "STEM" in gs:
        for m, metrics in gs["STEM"].items():
            if m in base_corr:
                if "answer_correctness" in metrics:
                    base_corr[m]["STEM"] = metrics["answer_correctness"]
                if "rouge_score" in metrics:
                    base_rouge[m]["STEM"] = metrics["rouge_score"]

    # fill TV
    if "TV" in gs:
        for m, metrics in gs["TV"].items():
            if m in base_corr:
                if "answer_correctness" in metrics:
                    base_corr[m]["TV"] = metrics["answer_correctness"]
                if "rouge_score" in metrics:
                    base_rouge[m]["TV"] = metrics["rouge_score"]

    # make DFs: (datasets x methods)
    df_corr = pd.DataFrame(base_corr).T  # methods x datasets
    df_corr = df_corr.T                  # datasets x methods
    df_rouge = pd.DataFrame(base_rouge).T
    df_rouge = df_rouge.T

    # local plotting helper (won't overwrite your global one)
    def _grouped_barplot_two_x(methods, df, ax, ylabel: str):
        LABEL_FONTSIZE  = 20
        TICK_FONTSIZE   = 20
        VALUE_FONTSIZE  = 5
        SPINE_WIDTH     = 1.2

        x_labels = ["Medical", "Novel", "STEM", "TV"]
        x = np.arange(len(x_labels))
        m = len(methods)
        total_width = 0.84
        bar_w = total_width / m
        left = x - total_width / 2

        bar_containers = []

        for i, method in enumerate(methods):
            vals = df[method].to_numpy(dtype=float)

            # use your global styling arrays
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
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_FONTSIZE,
                )

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), gridspec_kw={"wspace": 0.3})

    _grouped_barplot_two_x(METHODS_FOUR, df_corr, axes[0], "Answer Correctness")
    _grouped_barplot_two_x(METHODS_FOUR, df_rouge, axes[1], "ROUGE-L")

    # shared legend
    handles = []
    labels = []
    for m in METHODS_FOUR:
        master_idx = METHODS_EXP1.index(m)
        color = cmap(master_idx % cmap.N)
        hatch = hatches[master_idx % len(hatches)]
        patch = plt.Rectangle((0, 0), 1, 1,
                              facecolor=color,
                              edgecolor="black",
                              linewidth=0.6,
                              hatch=hatch)
        handles.append(patch)
        labels.append(m)

    fig.legend(
        handles,
        labels,
        ncol=4,
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=1200, bbox_inches="tight")
    plt.close(fig)



# --------------------------------------------------------------
# main
# --------------------------------------------------------------
def main() -> None:
    # single feature plots (non-gs)
    # make_feature_plot("input_tokens", OUT_INPUT_TOK, "Avg Input Tokens")
    # make_feature_plot("gen_latency_sec", OUT_GEN_LAT, "Avg gen Latency (s)")
    # make_feature_plot("retrieval_latency_sec", OUT_RETR_LAT, "Avg Retrieval Latency (s)")


    # pair for non-gs
    make_feature_plot_pair(
        "gen_latency_sec",
        "retrieval_latency_sec",
        OUT_LAT_PAIR,
        "Avg gen Latency (s)",
        "Avg Retrieval Latency (s)",
    )

    make_feature_plot_pair(
        feature_name_left="input_tokens",
        feature_name_right="output_tokens",
        outfile=OUT_TOK_PAIR,
        ylabel_left="Avg Input Tokens",
        ylabel_right="Avg Output Tokens",
    )

    # pair for gs: correctness + rouge
    make_gs_plot_pair(OUT_GS_PAIR)

    make_gs_plot_full(OUT_GS_FULL)

    # # if you ALSO want a single rouge-only plot:
    # # (reuse the gs collector but plot only rouge)
    # gs = collect_exp2_gs_metrics(BASE_DIR)
    # data = {m: {"STEM": float("nan"), "TV": float("nan")} for m in METHODS_EXP1}
    # for m, metrics in gs["STEM"].items():
    #     if "rouge_score" in metrics:
    #         data[m]["STEM"] = metrics["rouge_score"]
    # for m, metrics in gs["TV"].items():
    #     if "rouge_score" in metrics:
    #         data[m]["TV"] = metrics["rouge_score"]
    # df = pd.DataFrame(data).T
    # df = df.T
    # present_methods = [
    #     m for m in METHODS_EXP1
    #     if m in df.columns and not (pd.isna(df[m]["STEM"]) and pd.isna(df[m]["TV"]))
    # ]
    # if present_methods:
    #     fig, ax = plt.subplots(figsize=(10.5, 5.8))
    #     _grouped_barplot_two_x(present_methods, df[present_methods], ax, "rouge_score")
    #     ax.legend(
    #         ncol=3,
    #         fontsize=15,
    #         loc="upper center",
    #         bbox_to_anchor=(0.5, 1.1),
    #         frameon=False,
    #     )
    #     fig.tight_layout()
    #     OUT_ROUGE.parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(OUT_ROUGE, dpi=1200, bbox_inches="tight")
    #     plt.close(fig)

    make_graph_tokens_workspace_3panel(OUT_GRAPH_3_PANEL)

    make_apr_codebook_diff_from_first(OUT_APR_CODEBOOK_DIFF)


if __name__ == "__main__":
    main()



# python get_plots_for_exp2.py