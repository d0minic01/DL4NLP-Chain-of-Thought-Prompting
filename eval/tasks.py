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


def discover_fewshot_sets():
    """Scan fewshots_and_cots/ for benchmark directories containing questions.jsonl.

    Returns:
        dict: {benchmark_name: {"questions_path": str, "techniques": {technique_name: path}}}
    """
    benchmarks = {}
    for bench_dir in sorted(COTS_DIR.iterdir()):
        questions_path = bench_dir / "questions.jsonl"

        techniques = {}
        for f in sorted(bench_dir.iterdir()):
            if f.suffix == ".jsonl" and f.name != "questions.jsonl":
                techniques[f.stem] = str(f)

        benchmarks[bench_dir.name] = {
            "questions_path": str(questions_path),
            "techniques": techniques,
        }
    return benchmarks


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


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def configure_runs(benchmarks_cfg, cot_techniques_cfg, cot_files):
    """Build run configurations for each benchmark × technique combination.

    Args:
        benchmarks_cfg: benchmarks section from config
        cot_techniques_cfg: cot_techniques section from config
        cot_files: output from discover_fewshot_sets()
    """
    runs = []
    for bench_name, bench_cfg in benchmarks_cfg.items():
        bench_cot_key = bench_cfg.get("cot_benchmark", bench_name)

        if bench_cot_key not in cot_files:
            logger.warning(
                f"Benchmark '{bench_name}' has no few-shot files in fewshots_and_cots/{bench_cot_key}/"
            )
            continue

        bench_files = cot_files[bench_cot_key]
        questions_path = bench_files["questions_path"]

        for technique in bench_cfg.get("cot_techniques", []):
            if technique not in cot_techniques_cfg:
                logger.warning(
                    f"Technique '{technique}' not defined in cot_techniques config, skipping"
                )
                continue

            tech_cfg = cot_techniques_cfg[technique]
            tech_type = tech_cfg["type"]

            run = {
                "bench_name": f"{bench_name}_{technique}",
                "bench_base": bench_name,
                "task": bench_cfg["task"],
                "limit": bench_cfg.get("limit"),
                "technique": technique,
                "technique_type": tech_type,
                "questions_path": questions_path,
                "is_cot": tech_type != "none",
            }

            if tech_type == "none":
                pass  # no extra info needed

            elif tech_type == "zero_shot":
                run["instruction"] = tech_cfg.get(
                    "instruction", "Let's think step by step."
                )

            elif tech_type == "fewshot":
                technique_path = bench_files["techniques"].get(technique)
                if technique_path is None:
                    logger.warning(
                        f"Technique '{technique}' requested for '{bench_name}' "
                        f"but {bench_cot_key}/{technique}.jsonl not found, skipping"
                    )
                    continue
                run["technique_path"] = technique_path

            runs.append(run)

    return runs


def build_task_yaml(
    base_task,
    task_name,
    technique_type,
    questions_path,
    technique_path=None,
    instruction=None,
):
    """Build a task YAML for lm_eval.

    Args:
        base_task: benchmark task name (e.g., 'gsm8k')
        task_name: full run name (e.g., 'gsm8k_step_by_step')
        technique_type: 'none', 'zero_shot', or 'fewshot'
        questions_path: path to questions.jsonl
        technique_path: path to technique .jsonl (for fewshot type)
        instruction: zero-shot instruction string
    """
    yaml_path = BENCHMARKS_DIR / f"{base_task}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Benchmark YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        task_def = yaml.safe_load(f)

    task_def["task"] = task_name
    task_def.pop("tag", None)
    task_def.pop("tags", None)

    questions = load_jsonl(questions_path)
    num_fewshot = 0

    if technique_type == "none":
        task_def["num_fewshot"] = len(questions)
        task_def["fewshot_split"] = None
        fewshot_cfg = task_def.get("fewshot_config", {})
        fewshot_cfg["sampler"] = "first_n"
        fewshot_cfg["samples"] = questions
        fewshot_cfg["doc_to_target"] = "The answer is {{target}}."
        task_def["fewshot_config"] = fewshot_cfg
        num_fewshot = len(questions)

    elif technique_type == "zero_shot":
        task_def["num_fewshot"] = 0
        task_def["fewshot_split"] = None
        existing_text = task_def.get("doc_to_text", "")
        task_def["doc_to_text"] = f"{existing_text.rstrip()}\n{instruction}\n"

    elif technique_type == "fewshot":
        cot_records = load_jsonl(technique_path)
        if len(cot_records) != len(questions):
            logger.warning(
                f"Mismatch: {len(questions)} questions but {len(cot_records)} "
                f"CoT records in {technique_path}. Using min length."
            )
        n = min(len(questions), len(cot_records))
        fewshot_examples = []
        for i in range(n):
            fewshot_examples.append(
                {
                    "question": questions[i]["question"],
                    "chain_of_thought": cot_records[i]["chain_of_thought"],
                    "target": questions[i]["target"],
                }
            )

        task_def["num_fewshot"] = n
        task_def["fewshot_split"] = None
        fewshot_cfg = task_def.get("fewshot_config", {})
        fewshot_cfg["sampler"] = "first_n"
        fewshot_cfg["samples"] = fewshot_examples
        task_def["fewshot_config"] = fewshot_cfg
        num_fewshot = n

    return yaml.dump(task_def, default_flow_style=False, sort_keys=False), num_fewshot
