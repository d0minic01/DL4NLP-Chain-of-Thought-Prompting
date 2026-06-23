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
}

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
        lambda t: "baseline" if t == "fewshot_answer_only" else "cot"
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
                        fontsize=7,
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

    print(f"Saved plots to {out_dir}/")
    print(f"  benchmark_comparison: one PNG per model")
    print(f"  technique_comparison: one PNG per model")
    print(f"  cot_delta_heatmap.png")
    print(f"  scaling_curves.png")
    print(f"  category_summary.png")


if __name__ == "__main__":
    main()
