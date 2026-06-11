import argparse

from eval.tasks import load_config, resolve_models
from eval.runner import run_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run lm_eval benchmarks")
    parser.add_argument(
        "-b", "--benchmark",
        help="Run a single benchmark by name (from benchmarks.yaml)",
    )
    parser.add_argument(
        "-m", "--model",
        help="Run a single model by name (from benchmarks.yaml)",
    )
    parser.add_argument(
        "-c", "--cot",
        help="Run only a specific CoT variant by name (from benchmarks.yaml)",
    )
    return parser.parse_args()


def apply_filters(config, args):
    if args.benchmark:
        benchmarks = config.get("benchmarks", {})
        if args.benchmark not in benchmarks:
            available = ", ".join(benchmarks.keys())
            print(f"Benchmark '{args.benchmark}' not found. Available: {available}")
            raise SystemExit(1)
        config["benchmarks"] = {args.benchmark: benchmarks[args.benchmark]}

    if args.cot:
        cots = config.get("cots", {})
        if args.cot not in cots:
            available = ", ".join(cots.keys())
            print(f"CoT '{args.cot}' not found. Available: {available}")
            raise SystemExit(1)
        config["cots"] = {args.cot: cots[args.cot]}

    if args.benchmark and not args.cot:
        config["cots"] = {}

    if args.benchmark and args.cot:
        config["run_base"] = False

    if args.model:
        resolved = resolve_models(config.get("models", {}))
        if args.model not in resolved:
            available = ", ".join(resolved.keys())
            print(f"Model '{args.model}' not found. Available: {available}")
            raise SystemExit(1)
        config["models"] = {
            "defaults": config.get("models", {}).get("defaults", {}),
            args.model: {k: v for k, v in resolved[args.model].items() if k not in config["models"].get("defaults", {})},
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
    run_eval(config)
