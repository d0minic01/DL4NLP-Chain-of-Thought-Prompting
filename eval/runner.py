import gc
import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

from eval.tasks import (
    build_task_yaml,
    configure_runs,
    discover_prompt_sets,
    resolve_models,
)


def calculate_max_vram() -> float | None:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    return None


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_eval_kwargs(
    model_name, model_cfg, run, task_manager, device, log_samples=True
) -> dict:
    kwargs = dict(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=[run["bench_name"]],
        num_fewshot=run["num_fewshot"],
        log_samples=log_samples,
        task_manager=task_manager,
        device=device,
    )
    if run["limit"] is not None:
        kwargs["limit"] = run["limit"]
    batch_size = model_cfg.get("batch_size")
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return kwargs


def extract_accuracy(task_results):
    for key in [
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
        "acc_norm,none",
    ]:
        if key in task_results:
            return task_results[key], key
    if task_results:
        first_key = next(iter(task_results))
        return task_results[first_key], first_key
    return None, None


def save_result(results_dir, bench_base, model_name, row, cot_name=None, samples=None):

    safe_name = model_name.replace("/", "_").replace(":", "_")
    model_dir = results_dir / bench_base / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{safe_name}-{cot_name}.csv" if cot_name else f"{safe_name}.csv"
    csv_path = model_dir / csv_filename
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    if samples is not None:
        json_filename = (
            f"{safe_name}-{cot_name}-samples.json"
            if cot_name
            else f"{safe_name}-samples.json"
        )
        json_path = model_dir / json_filename
        with open(json_path, "w") as f:
            json.dump(samples, f, indent=2, default=str)


def run_eval(config):
    benchmarks = config.get("benchmarks", {})
    prompt_sets = discover_prompt_sets()
    models = resolve_models(config.get("models", {}))
    results_dir = Path("results")
    log_samples = config.get("log_samples", False)

    runs = configure_runs(benchmarks, prompt_sets)

    if config.get("cot_only"):
        runs = [r for r in runs if r["is_cot"]]
    elif config.get("baseline_only"):
        runs = [r for r in runs if not r["is_cot"]]

    if prompt_set_override := config.get("prompt_set_override"):
        for run in runs:
            run["prompt_cfg"] = prompt_sets[prompt_set_override]
            run["prompt_set"] = prompt_set_override

    if config.get("limit") is not None:
        for run in runs:
            run["limit"] = config["limit"]

    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdir:
        for run in runs:
            yaml_content, num_fewshot = build_task_yaml(
                run["task"],
                run["prompt_cfg"],
                run["bench_name"],
                run["is_cot"],
            )
            tmp_file = Path(tmpdir) / f"{run['bench_name']}.yaml"
            tmp_file.write_text(yaml_content)
            run["num_fewshot"] = num_fewshot

        task_manager = TaskManager(include_path=[tmpdir])

        all_results = []

        for model_name, model_cfg in models.items():
            for run in runs:
                bench_name = run["bench_name"]
                bench_base = run["bench_base"]
                prompt_set = run.get("prompt_set")

                clear_gpu()
                start = time.perf_counter()
                samples = None

                try:
                    eval_kwargs = build_eval_kwargs(
                        model_name,
                        model_cfg,
                        run,
                        task_manager,
                        device,
                        log_samples=log_samples,
                    )
                    result = simple_evaluate(**eval_kwargs)
                    elapsed = time.perf_counter() - start

                    if result is None:
                        raise RuntimeError("lm_eval returned no result")

                    results = result.get("results", {}).get(bench_name, {})

                    accuracy, metric_used = extract_accuracy(results)

                    samples = None
                    if log_samples:
                        samples = result.get("samples", {}).get(bench_name, [])

                    peak_vram_gb = calculate_max_vram()
                    print(f"    Result: {accuracy} ({metric_used}) in {elapsed:.1f}s")

                except Exception as e:
                    elapsed = time.perf_counter() - start
                    accuracy = None
                    metric_used = None
                    peak_vram_gb = None
                    print(f"    FAILED: {e}")

                row = {
                    "model": model_name,
                    "benchmark": bench_name,
                    "task": run["task"],
                    "accuracy": accuracy,
                    "metric": metric_used,
                    "num_fewshot": run.get("num_fewshot"),
                    "limit": run.get("limit"),
                    "eval_time_sec": round(elapsed, 2),
                    "peak_vram_gb": peak_vram_gb,
                }
                all_results.append(row)
                save_result(
                    results_dir,
                    bench_base,
                    model_name,
                    row,
                    prompt_set,
                    samples=samples,
                )

    clear_gpu()

    df = pd.DataFrame(all_results)

    return df
