"""Visualize evaluation results: baseline vs CoT accuracy per model/benchmark."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RESULTS_DIR = Path("results")

BENCHMARK_CATEGORIES = {
    "arithmetic": ["gsm8k", "asdiv", "mawps", "svamp", "aqua"],
    "commonsense": ["csqa", "strategyqa", "sports", "date"],
    "symbolic": ["coin_flip", "last_letter", "letter_shift"],
    "planning": ["saycan"],
}

# Human-readable technique labels
TECHNIQUE_DISPLAY = {
    "true_baseline": "No prompt",
    "think_step_by_step": "Think step-by-step",
    "fewshot_baseline": "Few-shot (answers only)",
    "paper_cot": "CoT (paper)",
    "contrastive_cot": "Contrastive CoT",
    "numbered_step_cot": "Numbered steps",
    "equation_only": "Equation only",
    "persona_cot": "Persona CoT",
    "caveman_mode": "Caveman",
}

# Consistent display order across all plots: simple baselines → CoT variants
TECHNIQUE_ORDER = [
    "true_baseline",
    "think_step_by_step",
    "fewshot_baseline",
    "paper_cot",
    "contrastive_cot",
    "numbered_step_cot",
    "equation_only",
    "persona_cot",
    "caveman_mode",
]

SIMPLE_TECHNIQUES = {"true_baseline", "think_step_by_step", "fewshot_baseline"}

# Approximate parameter counts (millions) for sorting and scatter plots
MODEL_PARAMS_M = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": 135,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 360,
    "google/gemma-3-270m-it": 270,
    "google/flan-t5-small": 80,
    "google/flan-t5-base": 250,
    "Qwen/Qwen2.5-0.5B-Instruct": 500,
    "google/flan-t5-large": 780,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 1700,
    "google/gemma-3-1b-it": 1000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1500,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 1500,
    "google/gemma-2-2b-it": 2000,
    "Qwen/Qwen2.5-3B-Instruct": 3000,
    "google/gemma-3-4b-it": 4000,
    "microsoft/Phi-3-mini-4k-instruct": 3800,
    "microsoft/Phi-4-mini-instruct": 3800,
    "Qwen/Qwen2.5-7B-Instruct": 7000,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 7000,
    "google/gemma-2-9b-it": 9000,
}


def load_results() -> pd.DataFrame:
    rows = []
    for csv_path in RESULTS_DIR.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if "bench_base" not in df.columns:
                df["bench_base"] = csv_path.parts[-3]
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def label_run_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'run_type' column: 'baseline' or 'cot'."""
    df = df.copy()
    df["run_type"] = df["technique_type"].map(
        lambda t: "baseline" if t == "fewshot_base" else "cot"
    )
    return df


def pivot_baseline_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table with one row per (model, bench_base), columns baseline & cot accuracy."""
    baseline = df[df["run_type"] == "baseline"][["model", "bench_base", "accuracy"]].rename(
        columns={"accuracy": "acc_baseline"}
    )
    cot = df[df["run_type"] != "baseline"][["model", "bench_base", "technique", "accuracy"]].rename(
        columns={"accuracy": "acc_cot"}
    )
    merged = baseline.merge(cot, on=["model", "bench_base"], how="outer")
    return merged


def plot_benchmark_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: baseline vs CoT for each benchmark, one chart per model."""
    if df.empty:
        return
    for model, mdf in df.groupby("model"):
        benches = sorted(mdf["bench_base"].unique())
        x = range(len(benches))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(benches) * 1.2), 5))
        baseline_accs = [mdf.loc[mdf["bench_base"] == b, "acc_baseline"].mean() for b in benches]
        cot_accs = [mdf.loc[mdf["bench_base"] == b, "acc_cot"].mean() for b in benches]

        ax.bar([i - width / 2 for i in x], baseline_accs, width, label="Baseline", color="#4C72B0")
        ax.bar([i + width / 2 for i in x], cot_accs, width, label="CoT", color="#DD8452")

        ax.set_xticks(list(x))
        ax.set_xticklabels(benches, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Baseline vs CoT — {model.split('/')[-1]}")
        ax.legend()
        fig.tight_layout()

        safe = model.replace("/", "_").replace(":", "_")
        fig.savefig(out_dir / f"{safe}_benchmark_comparison.png", dpi=150)
        plt.close(fig)


def plot_technique_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: all techniques side-by-side for each model."""
    if df.empty:
        return

    for model, mdf in df.groupby("model"):
        techniques = sorted(mdf["technique"].unique())
        n_techniques = len(techniques)
        technique_colors = dict(zip(techniques, plt.cm.Set2(range(n_techniques))))

        benches = sorted(mdf["bench_base"].unique())
        n_benches = len(benches)
        fig, ax = plt.subplots(figsize=(max(8, n_benches * 1.8), 5))

        bar_width = 0.8 / n_techniques
        x_base = range(n_benches)

        for i, tech in enumerate(techniques):
            accs = []
            for b in benches:
                subset = mdf[(mdf["bench_base"] == b) & (mdf["technique"] == tech)]
                accs.append(subset["accuracy"].mean() if not subset.empty else None)
            offset = (i - n_techniques / 2 + 0.5) * bar_width
            bars = ax.bar(
                [x + offset for x in x_base],
                [a if a is not None else 0 for a in accs],
                bar_width,
                label=tech,
                color=technique_colors[tech],
            )
            for bar, val in zip(bars, accs):
                if val is not None and val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.0%}",
                        ha="center",
                        va="bottom",
                        fontsize=max(5, min(7, bar_width * 40)),
                        rotation=90 if bar_width < 0.05 else 0,
                    )

        ax.set_xticks(list(x_base))
        ax.set_xticklabels(benches, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Technique Comparison — {model.split('/')[-1]}")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()

        safe = model.replace("/", "_").replace(":", "_")
        fig.savefig(out_dir / f"{safe}_technique_comparison.png", dpi=150)
        plt.close(fig)


def plot_cot_delta_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of CoT gain (acc_cot - acc_baseline) across models x benchmarks."""
    if df.empty:
        return
    df = df.copy()
    df["delta"] = df["acc_cot"] - df["acc_baseline"]
    pivot = df.pivot_table(index="model", columns="bench_base", values="delta", aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.split("/")[-1] for m in pivot.index])
    plt.colorbar(im, ax=ax, label="CoT gain (Δ accuracy)")
    ax.set_title("CoT Gain Heatmap (green = CoT helps, red = CoT hurts)")
    fig.tight_layout()
    fig.savefig(out_dir / "cot_delta_heatmap.png", dpi=150)
    plt.close(fig)


def plot_scaling_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter+line: avg accuracy vs model size, one line per category and run type."""
    if df.empty:
        return
    df = df.copy()
    df["params_m"] = df["model"].map(MODEL_PARAMS_M)
    df = df.dropna(subset=["params_m"])
    if df.empty:
        return

    all_categories = list(BENCHMARK_CATEGORIES.keys())
    fig, axes = plt.subplots(1, len(all_categories), figsize=(6 * len(all_categories), 5), sharey=True)
    if len(all_categories) == 1:
        axes = [axes]

    for ax, category in zip(axes, all_categories):
        benches = BENCHMARK_CATEGORIES[category]
        cat_df = df[df["bench_base"].isin(benches)]
        if cat_df.empty:
            ax.set_title(category)
            continue

        for run_type, label, color in [
            ("baseline", "Baseline", "#4C72B0"),
            ("cot", "CoT", "#DD8452"),
        ]:
            col = "acc_baseline" if run_type == "baseline" else "acc_cot"
            grp = cat_df.groupby("params_m")[col].mean().dropna().sort_index()
            if grp.empty:
                continue
            ax.scatter(grp.index, grp.values, color=color, zorder=5)
            ax.plot(grp.index, grp.values, color=color, label=label, linewidth=1.5)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}M"))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Parameters")
        ax.set_title(category.capitalize())
        ax.legend()

    axes[0].set_ylabel("Average Accuracy")
    fig.suptitle("Scaling Curves: Baseline vs CoT by Task Category", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_curves.png", dpi=150)
    plt.close(fig)


def plot_category_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: average CoT gain per category, per model family."""
    if df.empty:
        return
    df = df.copy()
    df["delta"] = df["acc_cot"] - df["acc_baseline"]

    category_map = {b: cat for cat, benches in BENCHMARK_CATEGORIES.items() for b in benches}
    df["category"] = df["bench_base"].map(category_map)
    df = df.dropna(subset=["category", "delta"])
    if df.empty:
        return

    summary = df.groupby("category")["delta"].mean().reindex(list(BENCHMARK_CATEGORIES.keys()))

    valid = summary.dropna()
    lo = min(valid.min() - 0.03, -0.01) if not valid.empty else -0.1
    hi = max(valid.max() + 0.05, 0.05) if not valid.empty else 0.1

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#55A868" if (not pd.isna(v) and v >= 0) else "#C44E52" for v in summary.values]
    bars = ax.bar(summary.index, summary.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Average CoT gain")
    ax.set_title("Average CoT Gain by Task Category")
    for bar, val in zip(bars, summary.values):
        if not pd.isna(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.005 if val >= 0 else -0.015),
                f"{val:+.1%}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_dir / "category_summary.png", dpi=150)
    plt.close(fig)


def _technique_colors(techniques: list[str]) -> dict[str, tuple]:
    """Assign consistent colors across plots, ordered by TECHNIQUE_ORDER."""
    palette = plt.cm.tab10.colors
    return {t: palette[i % len(palette)] for i, t in enumerate(TECHNIQUE_ORDER) if t in techniques}


def plot_technique_scaling_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-technique accuracy vs. model scale (reproduces paper Figs 2/3/4).

    Each line is one prompting technique; subplots split by task category.
    """
    if df.empty:
        return
    df = df.copy()
    df["params_m"] = df["model"].map(MODEL_PARAMS_M)
    df = df.dropna(subset=["params_m", "accuracy"])
    if df.empty:
        return

    present = [t for t in TECHNIQUE_ORDER if t in df["technique"].unique()]
    colors = _technique_colors(present)

    all_categories = list(BENCHMARK_CATEGORIES.keys())
    fig, axes = plt.subplots(1, len(all_categories), figsize=(6 * len(all_categories), 5), sharey=True)
    if len(all_categories) == 1:
        axes = [axes]

    for ax, category in zip(axes, all_categories):
        benches = BENCHMARK_CATEGORIES[category]
        cat_df = df[df["bench_base"].isin(benches)]
        if cat_df.empty:
            ax.set_title(category.capitalize())
            continue

        for tech in present:
            grp = (
                cat_df[cat_df["technique"] == tech]
                .groupby("params_m")["accuracy"]
                .mean()
                .dropna()
                .sort_index()
            )
            if grp.empty:
                continue
            linestyle = "-" if tech in SIMPLE_TECHNIQUES else "--"
            ax.scatter(grp.index, grp.values, color=colors[tech], zorder=5, s=40)
            ax.plot(
                grp.index,
                grp.values,
                color=colors[tech],
                label=TECHNIQUE_DISPLAY.get(tech, tech),
                linewidth=1.5,
                linestyle=linestyle,
            )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}M"))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Parameters")
        ax.set_title(category.capitalize())

    axes[0].set_ylabel("Average Accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.14, 0.5), fontsize=8)
    fig.suptitle("Accuracy vs. Model Scale by Prompting Technique\n(solid = simple baselines, dashed = CoT)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    fig.savefig(out_dir / "technique_scaling_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_sufficiency(df: pd.DataFrame, out_dir: Path) -> None:
    """Compare simple baselines vs CoT techniques by task category.

    Answers the question: 'Is think-step-by-step or few-shot already enough?'
    (Inspired by the ablation study in Wei et al. 2022, Figure 5.)
    """
    if df.empty:
        return

    category_map = {b: cat for cat, benches in BENCHMARK_CATEGORIES.items() for b in benches}
    df = df.copy()
    df["category"] = df["bench_base"].map(category_map)
    df = df.dropna(subset=["category", "accuracy"])
    if df.empty:
        return

    present = [t for t in TECHNIQUE_ORDER if t in df["technique"].unique()]
    grouped = (
        df.groupby(["category", "technique"])["accuracy"]
        .mean()
        .unstack("technique")
        .reindex(list(BENCHMARK_CATEGORIES.keys()))
    )
    grouped = grouped[[t for t in present if t in grouped.columns]]

    if grouped.empty:
        return

    # Color: grey shades for simple baselines, orange palette for CoT
    simple_colors = ["#95A5A6", "#5DADE2", "#2ECC71"]
    cot_colors = plt.cm.Oranges([0.45 + 0.45 * i / max(len(present) - len(SIMPLE_TECHNIQUES) - 1, 1)
                                  for i in range(len(present) - len(SIMPLE_TECHNIQUES))])
    colors = []
    simple_idx = 0
    cot_idx = 0
    for t in grouped.columns:
        if t in SIMPLE_TECHNIQUES:
            colors.append(simple_colors[simple_idx % len(simple_colors)])
            simple_idx += 1
        else:
            colors.append(cot_colors[cot_idx])
            cot_idx += 1

    n_cats = len(grouped)
    n_techs = len(grouped.columns)
    bar_width = 0.8 / n_techs

    fig, ax = plt.subplots(figsize=(max(10, n_cats * n_techs * 0.55), 5))
    x = range(n_cats)

    for i, (tech, color) in enumerate(zip(grouped.columns, colors)):
        offset = (i - n_techs / 2 + 0.5) * bar_width
        vals = grouped[tech].values
        ax.bar(
            [xi + offset for xi in x],
            [v if not pd.isna(v) else 0 for v in vals],
            bar_width,
            label=TECHNIQUE_DISPLAY.get(tech, tech),
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([c.capitalize() for c in grouped.index])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Is a Simple Baseline Already Enough?\n(grey/blue = simple baselines, orange = CoT techniques)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_sufficiency.png", dpi=150)
    plt.close(fig)


def plot_output_efficiency(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: average output length vs. accuracy per (model, technique).

    Shows which techniques spend more tokens and whether that investment pays off.
    """
    if df.empty:
        return
    df = df.copy().dropna(subset=["avg_output_chars", "accuracy"])
    if df.empty:
        return

    # Average across benchmarks so each point = one (model, technique) combo
    grouped = (
        df.groupby(["model", "technique"])[["avg_output_chars", "accuracy"]]
        .mean()
        .reset_index()
    )

    present = [t for t in TECHNIQUE_ORDER if t in grouped["technique"].unique()]
    colors = _technique_colors(present)

    fig, ax = plt.subplots(figsize=(8, 6))

    for tech in present:
        sub = grouped[grouped["technique"] == tech]
        ax.scatter(
            sub["avg_output_chars"],
            sub["accuracy"],
            label=TECHNIQUE_DISPLAY.get(tech, tech),
            color=colors[tech],
            s=70,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("Average Output Characters (proxy for token usage)")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Output Length vs. Accuracy by Technique\n(each point = one model)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "output_efficiency.png", dpi=150)
    plt.close(fig)


def plot_invalid_rate_by_model(df: pd.DataFrame, out_dir: Path) -> None:
    """Malformed / unparseable response rate per model, grouped by technique.

    Illustrates that small models often fail at format before they can reason.
    Models are ordered left-to-right by parameter count (small → large).
    """
    if df.empty:
        return
    df = df.copy().dropna(subset=["invalid_rate"])
    if df.empty:
        return

    # Sort models small → large
    df["params_m"] = df["model"].map(MODEL_PARAMS_M)
    models_sorted = (
        df.groupby("model")["params_m"]
        .first()
        .sort_values()
        .index.tolist()
    )
    unknown = sorted(m for m in df["model"].unique() if m not in models_sorted)
    models_sorted = models_sorted + unknown

    present = [t for t in TECHNIQUE_ORDER if t in df["technique"].unique()]
    colors = _technique_colors(present)

    n_models = len(models_sorted)
    n_techs = len(present)
    if n_models == 0 or n_techs == 0:
        return

    bar_width = 0.8 / n_techs
    fig, ax = plt.subplots(figsize=(max(10, n_models * n_techs * 0.45), 5))
    x = range(n_models)

    for i, tech in enumerate(present):
        tech_df = df[df["technique"] == tech]
        rates = [
            tech_df[tech_df["model"] == m]["invalid_rate"].mean()
            if not tech_df[tech_df["model"] == m].empty
            else None
            for m in models_sorted
        ]
        offset = (i - n_techs / 2 + 0.5) * bar_width
        ax.bar(
            [xi + offset for xi in x],
            [r if r is not None else 0 for r in rates],
            bar_width,
            label=TECHNIQUE_DISPLAY.get(tech, tech),
            color=colors[tech],
        )

    def _model_label(m: str) -> str:
        params = MODEL_PARAMS_M.get(m)
        suffix = f"\n({params}M)" if params else ""
        return m.split("/")[-1] + suffix

    ax.set_xticks(list(x))
    ax.set_xticklabels([_model_label(m) for m in models_sorted], rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Invalid Response Rate")
    ax.set_title("Malformed Answer Rate by Model & Technique\n(higher = model fails to follow the answer format)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "invalid_rate_by_model.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize CoT evaluation results")
    parser.add_argument(
        "-o", "--output", default="plots", help="Output directory for plots (default: plots/)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_results()
    if raw.empty:
        print("No result CSVs found in results/. Run evaluations first.")
        return

    df = label_run_type(raw)
    pivot = pivot_baseline_cot(df)

    plot_benchmark_comparison(pivot, out_dir)
    plot_technique_comparison(raw, out_dir)
    plot_cot_delta_heatmap(pivot, out_dir)
    plot_scaling_curves(pivot, out_dir)
    plot_category_summary(pivot, out_dir)
    plot_technique_scaling_curves(raw, out_dir)
    plot_baseline_sufficiency(raw, out_dir)
    plot_output_efficiency(raw, out_dir)
    plot_invalid_rate_by_model(raw, out_dir)

    print(f"Saved plots to {out_dir}/")
    print(f"  benchmark_comparison:      one PNG per model")
    print(f"  technique_comparison:      one PNG per model")
    print(f"  cot_delta_heatmap.png")
    print(f"  scaling_curves.png")
    print(f"  category_summary.png")
    print(f"  technique_scaling_curves.png  (paper Fig 2/3/4 — per-technique lines)")
    print(f"  baseline_sufficiency.png      (is step-by-step / few-shot already enough?)")
    print(f"  output_efficiency.png         (output length vs. accuracy trade-off)")
    print(f"  invalid_rate_by_model.png     (malformed answer rates, small-model formatting failures)")


if __name__ == "__main__":
    main()
