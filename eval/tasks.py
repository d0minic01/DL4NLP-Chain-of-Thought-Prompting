import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
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


def build_task_yaml(base_task, prompt_cfg, task_name, is_cot):
    yaml_path = BENCHMARKS_DIR / f"{base_task}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Benchmark YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        task_def = yaml.safe_load(f)

    task_def["task"] = task_name
    task_def.pop("tag", None)
    task_def.pop("tags", None)

    raw_examples = load_examples(prompt_cfg["path"])
    task_def["num_fewshot"] = len(raw_examples)
    task_def["fewshot_split"] = None

    fewshot_cfg = task_def.get("fewshot_config", {})
    fewshot_cfg["sampler"] = "first_n"
    fewshot_cfg["samples"] = raw_examples
    if not is_cot:
        fewshot_cfg["doc_to_target"] = "The answer is {{target}}."
    task_def["fewshot_config"] = fewshot_cfg

    return yaml.dump(task_def, default_flow_style=False, sort_keys=False), len(
        raw_examples
    )
