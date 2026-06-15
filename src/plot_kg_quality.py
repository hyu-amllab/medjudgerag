#!/usr/bin/env python3
"""
plot_kg_quality.py — Paper-ready figures for the
KG-quality-vs-λ_g ablation.

Reads JSON outputs of analyze_kg_quality.py for each
(backbone, benchmark) pair and produces four figures:

  (a) Accuracy bar chart       : X = (backbone-benchmark) category,
                                 5 bars per category (one per λ_g)
  (b) Orphan-rate bar chart    : same layout as (a)
  (c) Empty-rate bar chart     : same layout as (a)
  (d) KG length line plot      : X = λ_g, Y = avg #entities,
                                 one line per (backbone, benchmark)

Conventions:
  - Legend rendered inside each plot at top-right (no figure title).
  - For bar charts: bars within a (backbone-benchmark) group are colored
    by λ_g (sequential Blues colormap, light = small λ_g).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = PROJECT_DIR / "results" / "kg_quality"
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Path to GPT-synthesized gold KG targets (for reference lines in (d)).
GOLD_DATA_PATH = PROJECT_DIR / "data" / "teacher_traces_postprocessed.jsonl"

# Result JSONs follow the naming convention written by
# analyze_kg_quality.py.
JSON_TEMPLATE = "{backbone}_{benchmark}.json"

BACKBONES = ["llama3", "mistral"]
BENCHMARKS = ["medqa", "medmcqa"]
BENCHMARK_DISPLAY = {"medqa": "MedQA", "medmcqa": "MedMCQA"}
BACKBONE_DISPLAY = {"llama3": "Llama", "mistral": "Mistral"}

# Colors: backbone -> color
BACKBONE_COLORS = {
    "llama3": "#d62728",
    "mistral": "#1f77b4",
}

# Order requested by the user (used both in bar charts and line plot).
CATEGORIES = [
    ("mistral", "medqa"),
    ("mistral", "medmcqa"),
    ("llama3",  "medqa"),
    ("llama3",  "medmcqa"),
]

# Per-category color / marker / linestyle.
CATEGORY_STYLE = {
    ("mistral", "medqa"):   {"color": "#1f77b4", "marker": "o", "ls": "-"},
    ("mistral", "medmcqa"): {"color": "#aec7e8", "marker": "s", "ls": "--"},
    ("llama3",  "medqa"):   {"color": "#d62728", "marker": "^", "ls": "-"},
    ("llama3",  "medmcqa"): {"color": "#ff9896", "marker": "D", "ls": "--"},
}


def category_label(bb, bm):
    return f"{BACKBONE_DISPLAY[bb]} / {BENCHMARK_DISPLAY[bm]}"


# Unified legend styling so every figure renders the legend box at the same
# size/spacing regardless of the number of entries.
LEGEND_KWARGS = dict(
    frameon=True,
    framealpha=0.92,
    fontsize=9,
    handlelength=1.8,
    handletextpad=0.6,
    borderpad=0.5,
    labelspacing=0.45,
    handleheight=1.0,
)


# ── Loaders ───────────────────────────────────────────────────────
def load_one(backbone, benchmark):
    path = ANALYSIS_DIR / JSON_TEMPLATE.format(backbone=backbone, benchmark=benchmark)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    entries.sort(key=lambda e: e["lambda_g"])
    return entries


def load_all():
    """data[backbone][benchmark] = sorted entries list (by λ_g)."""
    data = {}
    for bb in BACKBONES:
        data[bb] = {}
        for bm in BENCHMARKS:
            entries = load_one(bb, bm)
            if entries is None:
                print(f"[warn] missing JSON for {bb}/{bm}")
                continue
            data[bb][bm] = entries
    return data


def lambda_axis(data):
    """Return the sorted unique λ_g values across all loaded series."""
    seen = set()
    for bb in data:
        for bm in data[bb]:
            for e in data[bb][bm]:
                seen.add(e["lambda_g"])
    return sorted(seen)


def metric_value(entries, lambda_g, key):
    for e in entries:
        if e["lambda_g"] == lambda_g:
            return e[key]
    return None


def compute_gold_avg_entities(benchmarks=None):
    """Average #entities in GPT-synthesized gold KGs, grouped by benchmark."""
    if not GOLD_DATA_PATH.exists():
        print(f"[warn] gold data file missing: {GOLD_DATA_PATH}")
        return {}

    bench_set = set(benchmarks) if benchmarks else None
    counts = {}
    with open(GOLD_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            bm = d.get("benchmark")
            if bench_set is not None and bm not in bench_set:
                continue
            kg = d.get("kg", "") or ""
            n_ents = len(re.findall(r'\(\s*"Entity"\s*,\s*"([^"]*)"', kg))
            counts.setdefault(bm, []).append(n_ents)

    return {bm: (sum(v) / len(v) if v else None) for bm, v in counts.items()}


# ── Style ─────────────────────────────────────────────────────────
def setup_paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, basename):
    pdf = FIG_DIR / f"{basename}.pdf"
    png = FIG_DIR / f"{basename}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"[saved] {pdf}")
    print(f"[saved] {png}")


# ── Bar-chart helper ──────────────────────────────────────────────
def grouped_bar_chart(data, metric_key, ylabel, savename, ylim=None):
    """Bar chart with X = λ_g, 4 category bars per λ_g position."""
    lambdas = lambda_axis(data)
    n_lambda = len(lambdas)
    n_cats = len(CATEGORIES)

    bar_width = 0.18
    inner_pad = 0.0
    group_gap = 0.6

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    group_centers = []
    cursor = 0.0
    for lg in lambdas:
        group_left = cursor
        for (bb, bm) in CATEGORIES:
            if bm not in data.get(bb, {}):
                cursor += bar_width + inner_pad
                continue
            val = metric_value(data[bb][bm], lg, metric_key)
            if val is None:
                cursor += bar_width + inner_pad
                continue
            style = CATEGORY_STYLE[(bb, bm)]
            ax.bar(
                cursor + bar_width / 2.0, val,
                width=bar_width,
                color=style["color"],
                edgecolor="white", linewidth=0.5,
            )
            cursor += bar_width + inner_pad
        group_right = cursor - inner_pad
        group_centers.append((group_left + group_right) / 2.0)
        cursor += group_gap

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{lg:.1f}" for lg in lambdas])
    ax.set_xlabel(r"KG loss weight ($\lambda_g$)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    if ylim is not None:
        ax.set_ylim(ylim)

    # Legend inside plot, top-right: 4 categories.
    legend_handles = [
        Patch(facecolor=CATEGORY_STYLE[(bb, bm)]["color"], edgecolor="white",
              label=category_label(bb, bm))
        for (bb, bm) in CATEGORIES
    ]
    ax.legend(handles=legend_handles, loc="upper right", **LEGEND_KWARGS)

    plt.tight_layout()
    save_fig(fig, savename)
    plt.close(fig)


# ── Bar-chart helper with broken Y-axis ──────────────────────────
def grouped_bar_chart_broken(data, metric_key, ylabel, savename,
                              bottom_ylim, top_ylim, height_ratios=(1, 2)):
    """Same layout as grouped_bar_chart but with a broken Y-axis to
    keep both the λ_g=0 outlier regime and the small λ_g≥0.1 regime
    visually readable.
    """
    lambdas = lambda_axis(data)

    bar_width = 0.18
    inner_pad = 0.0
    group_gap = 0.6

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 4.6),
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.06},
    )

    group_centers = []
    cursor = 0.0
    for lg in lambdas:
        group_left = cursor
        for (bb, bm) in CATEGORIES:
            if bm not in data.get(bb, {}):
                cursor += bar_width + inner_pad
                continue
            val = metric_value(data[bb][bm], lg, metric_key)
            if val is None:
                cursor += bar_width + inner_pad
                continue
            style = CATEGORY_STYLE[(bb, bm)]
            for ax in (ax_top, ax_bot):
                ax.bar(
                    cursor + bar_width / 2.0, val,
                    width=bar_width,
                    color=style["color"],
                    edgecolor="white", linewidth=0.5,
                )
            cursor += bar_width + inner_pad
        group_right = cursor - inner_pad
        group_centers.append((group_left + group_right) / 2.0)
        cursor += group_gap

    ax_top.set_ylim(top_ylim)
    ax_bot.set_ylim(bottom_ylim)

    # Hide spines that face each other; remove top axis's x-tick labels.
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # Diagonal break marks.
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False,
                  linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs["transform"] = ax_bot.transAxes
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # X-axis ticks and label on bottom axis.
    ax_bot.set_xticks(group_centers)
    ax_bot.set_xticklabels([f"{lg:.1f}" for lg in lambdas])
    ax_bot.set_xlabel(r"KG loss weight ($\lambda_g$)")

    # Shared Y-axis label centered between the two axes.
    fig.text(0.02, 0.5, ylabel, va="center", rotation="vertical", fontsize=11)

    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    # Legend on top axis (the upper-right corner there is typically empty
    # since outliers don't quite reach the very top of `top_ylim`).
    legend_handles = [
        Patch(facecolor=CATEGORY_STYLE[(bb, bm)]["color"], edgecolor="white",
              label=category_label(bb, bm))
        for (bb, bm) in CATEGORIES
    ]
    ax_top.legend(handles=legend_handles, loc="upper right",
                  frameon=True, framealpha=0.92)

    plt.tight_layout(rect=(0.04, 0, 1, 1))
    save_fig(fig, savename)
    plt.close(fig)


# ── Bar chart split into λ_g=0 panel + λ_g≥0.1 panel ─────────────
def grouped_bar_chart_split(data, metric_key, ylabel, savename,
                             outlier_lambda=0.0,
                             outlier_ylim=None, cluster_ylim=None,
                             width_ratios=(1, 4)):
    """Small-multiples version: λ_g=0 (outlier) and λ_g≥0.1 (cluster)
    are shown in two side-by-side panels with independent Y ranges.
    Within each panel, bars retain proportional encoding (height ∝ value),
    avoiding the misleading-encoding pitfalls of a broken Y-axis.
    """
    lambdas = lambda_axis(data)
    outlier_lambdas = [lg for lg in lambdas if lg == outlier_lambda]
    cluster_lambdas = [lg for lg in lambdas if lg != outlier_lambda]

    bar_width = 0.18
    inner_pad = 0.0
    group_gap = 0.6

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9.2, 4.2),
        gridspec_kw={"width_ratios": list(width_ratios), "wspace": 0.18},
    )

    def plot_bars(ax, lams):
        group_centers = []
        cursor = 0.0
        for lg in lams:
            group_left = cursor
            for (bb, bm) in CATEGORIES:
                if bm not in data.get(bb, {}):
                    cursor += bar_width + inner_pad
                    continue
                val = metric_value(data[bb][bm], lg, metric_key)
                if val is None:
                    cursor += bar_width + inner_pad
                    continue
                style = CATEGORY_STYLE[(bb, bm)]
                ax.bar(
                    cursor + bar_width / 2.0, val,
                    width=bar_width,
                    color=style["color"],
                    edgecolor="white", linewidth=0.5,
                )
                cursor += bar_width + inner_pad
            group_right = cursor - inner_pad
            group_centers.append((group_left + group_right) / 2.0)
            cursor += group_gap
        return group_centers

    # Left panel: outlier (λ_g = 0)
    centers_left = plot_bars(ax_left, outlier_lambdas)
    ax_left.set_xticks(centers_left)
    ax_left.set_xticklabels([f"{lg:.1f}" for lg in outlier_lambdas])
    if outlier_ylim is not None:
        ax_left.set_ylim(outlier_ylim)
    ax_left.set_ylabel(ylabel)
    ax_left.grid(axis="y", alpha=0.25, linestyle=":")

    # Right panel: cluster (λ_g ≥ 0.1)
    centers_right = plot_bars(ax_right, cluster_lambdas)
    ax_right.set_xticks(centers_right)
    ax_right.set_xticklabels([f"{lg:.1f}" for lg in cluster_lambdas])
    if cluster_ylim is not None:
        ax_right.set_ylim(cluster_ylim)
    ax_right.grid(axis="y", alpha=0.25, linestyle=":")

    # Shared X label across both panels.
    fig.supxlabel(r"KG loss weight ($\lambda_g$)", y=0.02)

    # Legend in the right panel's upper-right corner (where bars do not reach).
    legend_handles = [
        Patch(facecolor=CATEGORY_STYLE[(bb, bm)]["color"], edgecolor="white",
              label=category_label(bb, bm))
        for (bb, bm) in CATEGORIES
    ]
    ax_right.legend(handles=legend_handles, loc="upper right",
                    frameon=True, framealpha=0.92)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, savename)
    plt.close(fig)


# ── Drawing helpers (operate on a passed Axes) ───────────────────
def draw_grouped_bars(ax, data, metric_key, ylim=None,
                       bar_width=0.18, inner_pad=0.0, group_gap=0.3):
    """Draw grouped bars on a given Axes (for use in combined figures)."""
    lambdas = lambda_axis(data)
    group_centers = []
    cursor = 0.0
    for lg in lambdas:
        group_left = cursor
        for (bb, bm) in CATEGORIES:
            if bm not in data.get(bb, {}):
                cursor += bar_width + inner_pad
                continue
            val = metric_value(data[bb][bm], lg, metric_key)
            if val is None:
                cursor += bar_width + inner_pad
                continue
            style = CATEGORY_STYLE[(bb, bm)]
            ax.bar(
                cursor + bar_width / 2.0, val,
                width=bar_width,
                color=style["color"],
                edgecolor="white", linewidth=0.5,
            )
            cursor += bar_width + inner_pad
        group_right = cursor - inner_pad
        group_centers.append((group_left + group_right) / 2.0)
        cursor += group_gap

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{lg:.1f}" for lg in lambdas])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def draw_kg_length_lines(ax, data, ylim=None):
    """Draw the KG length line plot using evenly-spaced x positions
    (indices over λ_g) so that 0.0 and 0.1 don't overlap visually."""
    lambdas = lambda_axis(data)
    x_positions = list(range(len(lambdas)))  # evenly spaced

    for (bb, bm) in CATEGORIES:
        entries = data.get(bb, {}).get(bm)
        if not entries:
            continue
        # Map each entry's λ_g to its index in `lambdas`.
        ys = [None] * len(lambdas)
        for e in entries:
            if e["lambda_g"] in lambdas:
                idx = lambdas.index(e["lambda_g"])
                ys[idx] = e["avg_n_ents"]
        plot_xs = [x for x, y in zip(x_positions, ys) if y is not None]
        plot_ys = [y for y in ys if y is not None]

        style = CATEGORY_STYLE[(bb, bm)]
        ax.plot(
            plot_xs, plot_ys,
            color=style["color"], marker=style["marker"], linestyle=style["ls"],
            linewidth=1.6, markersize=6,
        )

    gold_avgs = compute_gold_avg_entities(benchmarks=BENCHMARKS)
    gold_styles = {
        "medqa":   {"color": "#444444", "ls": "-"},
        "medmcqa": {"color": "#444444", "ls": "--"},
    }
    for bm in BENCHMARKS:
        avg = gold_avgs.get(bm)
        if avg is None:
            continue
        s = gold_styles.get(bm)
        ax.axhline(avg, color=s["color"], linestyle=s["ls"],
                   linewidth=1.4, alpha=0.85)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{lg:.1f}" for lg in lambdas])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.25, linestyle=":")


# ── (d) line plot for KG length ───────────────────────────────────
def line_chart_kg_length(data, savename, ylim=None):
    lambdas = lambda_axis(data)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for (bb, bm) in CATEGORIES:
        entries = data.get(bb, {}).get(bm)
        if not entries:
            continue
        xs = [e["lambda_g"] for e in entries]
        ys = [e["avg_n_ents"] for e in entries]
        style = CATEGORY_STYLE[(bb, bm)]
        ax.plot(
            xs, ys,
            color=style["color"], marker=style["marker"], linestyle=style["ls"],
            linewidth=1.6, markersize=6,
            label=category_label(bb, bm),
        )

    # GPT gold reference lines: one horizontal line per benchmark.
    gold_avgs = compute_gold_avg_entities(benchmarks=BENCHMARKS)
    gold_styles = {
        "medqa":   {"color": "#444444", "ls": "-",  "label": "Gold / MedQA"},
        "medmcqa": {"color": "#444444", "ls": "--", "label": "Gold / MedMCQA"},
    }
    for bm in BENCHMARKS:
        avg = gold_avgs.get(bm)
        if avg is None:
            continue
        s = gold_styles.get(bm, {"color": "gray", "ls": ":", "label": f"Gold / {bm}"})
        ax.axhline(avg, color=s["color"], linestyle=s["ls"],
                   linewidth=1.4, alpha=0.85, label=s["label"])

    ax.set_xlabel(r"KG loss weight ($\lambda_g$)")
    ax.set_ylabel("Avg. entities per KG")
    ax.set_xticks(lambdas)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="upper right", **LEGEND_KWARGS)

    plt.tight_layout()
    save_fig(fig, savename)
    plt.close(fig)


# ── v5: combined 1x4 figure with shared bottom legend ───────────
def fig_combined_v5(data, savename):
    """Single 1x4 figure containing (a)(b)(c)(d), shared bottom legend.

    Designed for full-width (two-column) placement in a 2-column paper.
    Larger fonts, tight bar spacing, and a shared 6-entry legend below
    the four panels (4 model lines + 2 gold reference lines).
    """
    rc = {
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1, 4, figsize=(18.0, 5.0),
            gridspec_kw={"wspace": 0.35},
        )

        # (a) Accuracy
        draw_grouped_bars(axes[0], data, "answer_acc", ylim=(0, 72),
                           bar_width=0.52, group_gap=0.45)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_xlabel(r"KG loss weight ($\lambda_g$)")

        # (b) Orphan-relation rate
        draw_grouped_bars(axes[1], data, "orphan_rel_rate", ylim=(0, 78),
                           bar_width=0.52, group_gap=0.45)
        axes[1].set_ylabel("Orphan-relation rate (%)")
        axes[1].set_xlabel(r"KG loss weight ($\lambda_g$)")

        # (c) Empty-KG rate
        draw_grouped_bars(axes[2], data, "empty_rate", ylim=(0, 110),
                           bar_width=0.52, group_gap=0.45)
        axes[2].set_ylabel("Empty-KG rate (%)")
        axes[2].set_xlabel(r"KG loss weight ($\lambda_g$)")

        # (d) Avg #entities (KG length)
        draw_kg_length_lines(axes[3], data, ylim=(0, 38))
        axes[3].set_ylabel("Avg. entities per KG")
        axes[3].set_xlabel(r"KG loss weight ($\lambda_g$)")

        # Shared bottom legend (6 entries: 4 model categories + 2 gold).
        legend_handles = [
            Patch(facecolor=CATEGORY_STYLE[(bb, bm)]["color"], edgecolor="white",
                  label=category_label(bb, bm))
            for (bb, bm) in CATEGORIES
        ]
        legend_handles += [
            Line2D([], [], color="#444444", linestyle="-", linewidth=1.6,
                   label="Gold / MedQA"),
            Line2D([], [], color="#444444", linestyle="--", linewidth=1.6,
                   label="Gold / MedMCQA"),
        ]
        # Direct axes positioning so the legend can sit "almost touching"
        # the y-axis top spine. Tweak `top` (axes top in fig coords) and the
        # legend's `bbox_to_anchor.y` together — gap = bbox.y - top.
        plt.subplots_adjust(
            top=0.92,
            bottom=0.16,
            left=0.045,
            right=0.995,
            wspace=0.32,
        )

        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=6,
            frameon=True,
            bbox_to_anchor=(0.5, 0.925),  # ~0.005 above axes top → almost touching
            fontsize=14,
            handlelength=2.2,
            handletextpad=0.6,
            columnspacing=1.4,
            borderpad=0.7,
        )
        frame = legend.get_frame()
        frame.set_edgecolor("#cccccc")
        frame.set_facecolor("white")
        frame.set_linewidth(0.6)
        save_fig(fig, savename)
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────
def main():
    setup_paper_style()
    data = load_all()
    if not any(data[bb] for bb in BACKBONES):
        raise SystemExit(f"No JSON results found in {ANALYSIS_DIR}")

    # (a) Answer accuracy
    grouped_bar_chart(
        data,
        metric_key="answer_acc",
        ylabel="Accuracy (%)",
        savename="fig_a_accuracy_bars",
    )

    # (b) Orphan-relation rate
    grouped_bar_chart(
        data,
        metric_key="orphan_rel_rate",
        ylabel="Orphan-relation rate (%)",
        savename="fig_b_orphan_rate_bars",
    )

    # (c) Empty-KG rate
    grouped_bar_chart(
        data,
        metric_key="empty_rate",
        ylabel="Empty-KG rate (%)",
        savename="fig_c_empty_rate_bars",
    )

    # (b)-v2 Orphan rate with broken Y-axis (λ_g=0 regime split from λ_g≥0.1)
    grouped_bar_chart_broken(
        data,
        metric_key="orphan_rel_rate",
        ylabel="Orphan-relation rate (%)",
        savename="fig_b_orphan_rate_bars_v2",
        bottom_ylim=(0, 10),
        top_ylim=(15, 75),
        height_ratios=(1, 2),
    )

    # (c)-v2 Empty-KG rate with broken Y-axis
    grouped_bar_chart_broken(
        data,
        metric_key="empty_rate",
        ylabel="Empty-KG rate (%)",
        savename="fig_c_empty_rate_bars_v2",
        bottom_ylim=(0, 15),
        top_ylim=(18, 102),
        height_ratios=(1, 2),
    )

    # (b)-v3 / (c)-v3 small-multiples split (λ_g=0 vs λ_g≥0.1) — replaces
    # the broken-axis v2; preserves proportional bar encoding within each panel.
    grouped_bar_chart_split(
        data,
        metric_key="orphan_rel_rate",
        ylabel="Orphan-relation rate (%)",
        savename="fig_b_orphan_rate_bars_v3",
        outlier_ylim=(0, 75),
        cluster_ylim=(0, 10),
        width_ratios=(1, 4),
    )
    grouped_bar_chart_split(
        data,
        metric_key="empty_rate",
        ylabel="Empty-KG rate (%)",
        savename="fig_c_empty_rate_bars_v3",
        outlier_ylim=(0, 105),
        cluster_ylim=(0, 15),
        width_ratios=(1, 4),
    )

    # (d) Avg #entities (KG length proxy) as line plot
    line_chart_kg_length(data, savename="fig_d_kg_length_lines")

    # ── v4: legend overlap fix + unified legend styling across (a)-(d) ──
    grouped_bar_chart(
        data, metric_key="answer_acc", ylabel="Accuracy (%)",
        savename="fig_a_accuracy_bars_v4",
        ylim=(0, 95),
    )
    grouped_bar_chart(
        data, metric_key="orphan_rel_rate",
        ylabel="Orphan-relation rate (%)",
        savename="fig_b_orphan_rate_bars_v4",
        ylim=(0, 92),
    )
    grouped_bar_chart(
        data, metric_key="empty_rate",
        ylabel="Empty-KG rate (%)",
        savename="fig_c_empty_rate_bars_v4",
        ylim=(0, 122),
    )
    line_chart_kg_length(
        data, savename="fig_d_kg_length_lines_v4",
        ylim=(0, 44),
    )

    # ── v5: combined 1x4 figure with shared bottom legend (paper-ready) ──
    fig_combined_v5(data, savename="fig_kg_quality_v5_combined")


if __name__ == "__main__":
    main()
