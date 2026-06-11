import json
from pathlib import Path

import yaml


def load_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "benchmarks.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


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


def load_fewshot_examples(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def expand_runs(benchmarks, cots, run_base=True):
    runs = []
    for name, cfg in benchmarks.items():
        if not cfg.get("enabled", False):
            continue
        if run_base:
            runs.append({
                "bench_name": name,
                "bench_base": name,
                "task": cfg["task"],
                "num_fewshot": cfg.get("num_fewshot", 0),
                "limit": cfg.get("limit"),
                "is_cot": False,
            })
        for cot_name, cot_cfg in cots.items():
            runs.append({
                "bench_name": f"{name}_{cot_name}",
                "bench_base": name,
                "task": cfg["task"],
                "num_fewshot": cot_cfg.get("num_fewshot", 8),
                "limit": cfg.get("limit"),
                "is_cot": True,
                "cot_name": cot_name,
                "cot_cfg": cot_cfg,
                "cot_overrides": cfg.get("cot_overrides", {}),
            })
    return runs


def build_cot_task_yaml(base_task, cot_cfg, bench_overrides, base_tm):
    entry = base_tm._index[base_task]
    task_def = dict(entry.cfg)

    task_def["task"] = f"{base_task}_cot"
    task_def.pop("tag", None)
    task_def.pop("tags", None)

    examples = load_fewshot_examples(cot_cfg["fewshot_path"])
    task_def["num_fewshot"] = cot_cfg.get("num_fewshot", 8)
    task_def["fewshot_config"] = {"sampler": "first_n", "samples": examples}
    task_def["fewshot_split"] = None

    for key in ["output_type", "doc_to_text", "generation_kwargs", "filter_list"]:
        if key in bench_overrides:
            task_def[key] = bench_overrides[key]

    return yaml.dump(task_def, default_flow_style=False, sort_keys=False)
