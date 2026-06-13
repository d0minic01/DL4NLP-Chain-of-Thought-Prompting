import argparse
import os
import textwrap

from eval.runner import run_eval
from eval.tasks import discover_prompt_sets, load_config, resolve_models


def build_help_text():
    config = load_config()
    benchmarks = list(config.get("benchmarks", {}).keys())
    prompt_sets = list(discover_prompt_sets().keys())
    models = list(resolve_models(config.get("models", {})).keys())

    width = os.get_terminal_size().columns
    lines = ["The following are available:"]
    lines.append("\n")
    for label, items in [
        ("Benchmarks", benchmarks),
        ("Prompt sets", prompt_sets),
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
        action="store_true",
        help="Run the CoT variant (default: run the baseline fewshot variant)",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        metavar="PROMPT_SET",
        help="Override the prompt set (default: uses the one defined in benchmarks.yaml)",
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

    if args.prompts:
        prompt_sets = discover_prompt_sets()
        get_or_exit("Prompt set", args.prompts, prompt_sets)
        config["prompt_set_override"] = args.prompts

    if args.cot:
        config["cot_only"] = True
    elif args.benchmark:
        config["baseline_only"] = True

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


def main():
    args = parse_args()

    config = load_config()
    config = apply_filters(config, args)
    if args.debug:
        config["log_samples"] = True
    if args.limit is not None:
        config["limit"] = args.limit
    if args.verbose or args.output_file:
        config["verbose"] = True
    if args.output_file:
        config["output_file"] = args.output_file
    run_eval(config)
