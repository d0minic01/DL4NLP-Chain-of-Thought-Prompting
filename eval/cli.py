import argparse
import os
import textwrap

from eval.runner import run_eval
from eval.tasks import load_config, resolve_models


def build_help_text():
    config = load_config()
    benchmarks = list(config.get("benchmarks", {}).keys())
    cots = list(config.get("cots", {}).keys())
    models = list(resolve_models(config.get("models", {})).keys())

    width = os.get_terminal_size().columns
    lines = ["The following are available:"]
    lines.append("\n")
    for label, items in [
        ("Benchmarks", benchmarks),
        ("Cots", cots),
        ("Models", models),
    ]:
        text = f"\033[1m{label}\033[0m: {', '.join(items)}"
        lines.append(textwrap.fill(text, width=width, subsequent_indent="  "))
        lines.append("\n")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lm_eval benchmarks",
        epilog=build_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Run a single benchmark by name",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Run a single model by name",
    )
    parser.add_argument(
        "-c",
        "--cot",
        help="Run only a specific CoT variant by name",
    )
    parser.add_argument(
        "-l",
        "--log_samples",
        action="store_true",
        help="Log model inputs/outputs to JSON files",
    )
    return parser.parse_args()


def require_key(name, value, collection):
    if value not in collection:
        available = ", ".join(collection.keys())
        print(f"{name} '{value}' not found. Available: {available}")
        raise SystemExit(1)
    return collection[value]


def apply_filters(config, args):
    if args.benchmark:
        benchmarks = config.get("benchmarks", {})
        require_key("Benchmark", args.benchmark, benchmarks)
        config["benchmarks"] = {args.benchmark: benchmarks[args.benchmark]}

    if args.cot:
        cots = config.get("cots", {})
        require_key("CoT", args.cot, cots)
        config["cots"] = {args.cot: cots[args.cot]}

    if args.benchmark and not args.cot:
        # Keep the CoT referenced by the benchmark
        benchmarks = config.get("benchmarks", {})
        cot_name = benchmarks.get(args.benchmark, {}).get("cot")
        if cot_name:
            cots = config.get("cots", {})
            config["cots"] = {cot_name: cots[cot_name]} if cot_name in cots else {}

    if args.benchmark and args.cot:
        config["run_base"] = False

    if args.model:
        resolved = resolve_models(config.get("models", {}))
        require_key("Model", args.model, resolved)
        config["models"] = {
            "defaults": config.get("models", {}).get("defaults", {}),
            args.model: {
                k: v
                for k, v in resolved[args.model].items()
                if k not in config["models"].get("defaults", {})
            },
        }

    return config


def main():
    args = parse_args()

    any_flag = args.benchmark or args.model or args.cot
    if any_flag and not args.model:
        print("Error: -m/--model is required when using other flags")
        raise SystemExit(1)

    config = load_config()
    config = apply_filters(config, args)
    if args.log_samples:
        config["log_samples"] = True
    run_eval(config)
