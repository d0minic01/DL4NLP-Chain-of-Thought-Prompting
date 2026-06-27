import argparse
import os
import textwrap

from eval.runner import run_eval
from eval.tasks import load_config, resolve_models


def build_help_text():
    config = load_config()
    benchmarks = list(config.get("benchmarks", {}).keys())
    techniques = list(config.get("cot_techniques", {}).keys())
    models = list(resolve_models(config.get("models", {})).keys())

    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    lines = ["The following are available:"]
    lines.append("\n")
    for label, items in [
        ("Benchmarks", benchmarks),
        ("CoT Techniques", techniques),
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
        "-t",
        "--technique",
        metavar="TECHNIQUE",
        default=None,
        help="CoT technique to run (default: all from benchmarks.yaml)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Log model inputs/outputs to JSON files",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Override the number of examples to evaluate per benchmark",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each sample's prompt, model output, and extracted answer during evaluation",
    )
    parser.add_argument(
        "--output-file",
        metavar="FILE",
        help="Write verbose sample output to FILE instead of stdout (implies -v)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Quick smoke test: 2 examples, one benchmark per category "
            "(gsm8k / csqa / coin_flip / saycan), all techniques, smallest active model, "
            "debug output enabled. Completes in ~2–5 minutes."
        ),
    )
    return parser.parse_args()


def get_or_exit(name, value, collection):
    if value not in collection:
        available = ", ".join(collection.keys())
        print(f"{name} '{value}' not found. Available: {available}")
        raise SystemExit(1)
    return collection[value]


def apply_filters(config, args):
    if args.benchmark:
        benchmarks = config.get("benchmarks", {})
        get_or_exit("Benchmark", args.benchmark, benchmarks)
        config["benchmarks"] = {args.benchmark: benchmarks[args.benchmark]}

    available = config.get("cot_techniques", {})
    if args.technique is not None:
        if args.technique not in available:
            print(f"Technique '{args.technique}' not found. Available: {', '.join(available.keys())}")
            raise SystemExit(1)
        config["technique_override"] = [args.technique]

    if args.model:
        resolved = resolve_models(config.get("models", {}))
        get_or_exit("Model", args.model, resolved)
        config["models"] = {
            "defaults": config.get("models", {}).get("defaults", {}),
            args.model: {
                k: v
                for k, v in resolved[args.model].items()
                if k not in config["models"].get("defaults", {})
            },
        }

    return config


def apply_smoke(config: dict) -> dict:
    """Restrict config to a minimal subset for quick end-to-end validation."""
    # One benchmark per category to exercise every task type
    SMOKE_BENCHMARKS = {"gsm8k", "csqa", "coin_flip", "saycan"}
    config["benchmarks"] = {
        k: v
        for k, v in config.get("benchmarks", {}).items()
        if k in SMOKE_BENCHMARKS
    }

    # Use only the first (smallest) active model
    models = config.get("models", {})
    defaults = models.get("defaults", {})
    active = [k for k in models if k != "defaults"]
    if active:
        first = active[0]
        config["models"] = {"defaults": defaults, first: models[first]}

    config["limit"] = 2
    config["log_samples"] = True
    return config


def main():
    args = parse_args()

    config = load_config()
    config = apply_filters(config, args)

    if args.smoke:
        config = apply_smoke(config)

    if args.debug:
        config["log_samples"] = True
    if args.limit is not None:
        config["limit"] = args.limit
    if args.verbose or args.output_file:
        config["verbose"] = True
    if args.output_file:
        config["output_file"] = args.output_file
    run_eval(config)
