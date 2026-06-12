import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


COTS_DIR = Path(__file__).resolve().parent.parent / "fewshots_and_cots"


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "benchmarks.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def discover_prompt_sets():
    return {p.stem: {"path": str(p)} for p in sorted(COTS_DIR.glob("*.jsonl"))}


def resolve_models(models_cfg):
    defaults = models_cfg.get("defaults", {})
    resolved = {}
    for name, cfg in models_cfg.items():
        if name == "defaults":
            continue
        if cfg is None:
            cfg = {}
        resolved[name] = {**defaults, **cfg}
    return resolved


def resolve_path(cot_path):
    if cot_path.startswith("~"):
        return Path(cot_path).expanduser()
    return Path(cot_path)


def load_examples(fewshot_path):
    fewshot_path = resolve_path(fewshot_path)
    if not fewshot_path.exists():
        logger.warning(f"CoT path does not exist: {fewshot_path}")
        return []

    examples = []
    with open(fewshot_path) as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                doc = {k.replace("-", "_"): v for k, v in doc.items()}
                examples.append(doc)
    return examples


def extract_accuracy(task_results):
    """Extract accuracy and metric name from task results."""
    if not task_results:
        return None, None

    for metric in ["acc", "acc_norm", "f1"]:
        # lm_eval 0.4+ uses keys like "acc,none" instead of "acc"
        for key in task_results:
            if key == metric or key.startswith(f"{metric},"):
                return task_results[key], metric

    return None, None


def configure_runs(benchmarks, prompt_sets):
    runs = []
    for name, cfg in benchmarks.items():
        prompt_set_name = cfg.get("prompt_set")
        if prompt_set_name not in prompt_sets:
            raise ValueError(
                f"Benchmark '{name}' references prompt set '{prompt_set_name}' but it's not defined in prompt_sets. "
                f"Available prompt sets: {list(prompt_sets.keys())}"
            )
        prompt_cfg = prompt_sets[prompt_set_name]

        runs.append(
            {
                "bench_name": f"{name}_no_cot",
                "bench_base": name,
                "task": cfg["task"],
                "num_fewshot": cfg.get("num_fewshot", 0),
                "limit": cfg.get("limit"),
                "is_cot": False,
                "prompt_cfg": prompt_cfg,
            }
        )

        runs.append(
            {
                "bench_name": f"{name}_{prompt_set_name}",
                "bench_base": name,
                "task": cfg["task"],
                "num_fewshot": 0,
                "limit": cfg.get("limit"),
                "is_cot": True,
                "prompt_set": prompt_set_name,
                "prompt_cfg": prompt_cfg,
            }
        )

    return runs


def build_task_yaml(base_task, prompt_cfg, base_tm, task_name, is_cot):
    entry = base_tm._index[base_task]
    task_def = dict(entry.cfg)

    task_def["task"] = task_name
    task_def.pop("tag", None)
    task_def.pop("tags", None)

    raw_examples = load_examples(prompt_cfg["path"])
    task_def["num_fewshot"] = len(raw_examples)
    task_def["fewshot_split"] = None

    is_multiple_choice = task_def.get("output_type") == "multiple_choice"

    if is_multiple_choice:
        # Multiple choice tasks have choices embedded in our fewshot question strings.
        # Override doc_to_text to use just {{question}} so fewshot examples render correctly.
        # For test items the dataset question field lacks choices, so reconstruct them inline.
        task_def["doc_to_choice"] = None
        task_def["output_type"] = "generate_until"
        task_def["doc_to_text"] = "{{question}}\nAnswer:"
    else:
        task_def.pop("doc_to_choice", None)
        # Guard against fields our fewshot examples don't have (e.g. asdiv uses `body` + `question`)
        task_def["doc_to_text"] = re.sub(
            r"\{\{\s*(\w+)\s*\}\}",
            lambda m: f"{{{{{m.group(1)} if {m.group(1)} is defined else ''}}}}",
            task_def.get("doc_to_text", ""),
        )

    examples = [
        {
            "question": ex["question"],
            "target": (
                f"{ex['chain_of_thought']} The answer is {ex['target']}."
                if is_cot
                else f"The answer is {ex['target']}."
            ),
        }
        for ex in raw_examples
    ]

    task_def["fewshot_config"] = {
        "sampler": "first_n",
        "samples": examples,
        "doc_to_target": "{{target}}",
    }

    task_def["generation_kwargs"] = {
        "until": ["Q:", "Question:", "</s>", "<|im_end|>"],
        "do_sample": False,
        "temperature": 0.0,
    }
    task_def["metric_list"] = [
        {
            "metric": "exact_match",
            "aggregation": "mean",
            "higher_is_better": True,
            "ignore_case": True,
            "ignore_punctuation": False,
            "regexes_to_ignore": [",", "\\$"],
        }
    ]
    task_def["filter_list"] = [
        {
            "name": "strict-match",
            "filter": [
                {
                    "function": "regex",
                    "regex_pattern": r"The answer is (-?[0-9][0-9,]*(?:\.[0-9]+)?)",
                },
                {"function": "take_first"},
            ],
        },
        {
            "name": "flexible-extract",
            "filter": [
                {
                    "function": "regex",
                    "regex_pattern": r"(-?[$0-9.,]{2,})|(-?[0-9]+)",
                    "group_select": -1,
                },
                {"function": "take_first"},
            ],
        },
    ]

    return yaml.dump(task_def, default_flow_style=False, sort_keys=False), len(
        raw_examples
    )
