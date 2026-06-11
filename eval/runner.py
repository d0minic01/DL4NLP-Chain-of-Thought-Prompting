import gc
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

from eval.tasks import build_cot_task_yaml, expand_runs, resolve_models

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


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


def is_hf_type(model_type):
    return model_type.startswith("hf") or model_type == "steered"


def build_model_args(model_name, model_cfg):
    if is_hf_type(model_cfg["type"]):
        parts = [f"pretrained={model_name}"]
    else:
        parts = [f"model={model_name}"]
        if "base_url" in model_cfg:
            parts.append(f"base_url={model_cfg['base_url']}")
    return ",".join(parts)


def build_eval_kwargs(model_name, model_cfg, run, task_manager, device) -> dict:
    kwargs = dict(
        model=model_cfg["type"],
        model_args=build_model_args(model_name, model_cfg),
        tasks=[run["task"]],
        num_fewshot=run["num_fewshot"],
        log_samples=False,
        task_manager=task_manager,
    )
    if run["limit"] is not None:
        kwargs["limit"] = run["limit"]
    batch_size = model_cfg.get("batch_size")
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if not is_hf_type(model_cfg["type"]):
        kwargs["apply_chat_template"] = model_cfg.get("apply_chat_template", True)
        kwargs["fewshot_as_multiturn"] = model_cfg.get("fewshot_as_multiturn", False)
    if is_hf_type(model_cfg["type"]):
        kwargs["device"] = device
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


def save_result(results_dir, bench_base, model_name, row, cot_name=None):
    safe_name = model_name.replace("/", "_").replace(":", "_")
    model_dir = results_dir / bench_base / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)
    if cot_name:
        csv_path = model_dir / f"{safe_name}-{cot_name}.csv"
    else:
        csv_path = model_dir / f"{safe_name}.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)


def run_eval(config):
    benchmarks = config.get("benchmarks", {})
    cots = config.get("cots", {})
    models = resolve_models(config.get("models", {}))
    results_dir = Path("results")

    runs = expand_runs(benchmarks, cots, run_base=config.get("run_base", True))

    if not runs:
        print("No benchmarks enabled. Edit benchmarks.yaml to enable some.")
        return

    if not models:
        print("No models defined. Edit benchmarks.yaml to add models.")
        return

    device = get_device()

    include_paths = []
    if TASKS_DIR.exists():
        include_paths.append(str(TASKS_DIR))

    base_tm = TaskManager(include_path=include_paths)

    with tempfile.TemporaryDirectory() as tmpdir:
        for run in runs:
            if run["is_cot"]:
                yaml_content = build_cot_task_yaml(
                    run["task"], run["cot_cfg"], run["cot_overrides"], base_tm
                )
                tmp_file = Path(tmpdir) / f"{run['bench_name']}.yaml"
                tmp_file.write_text(yaml_content)
                run["task"] = f"{run['task']}_cot"

        task_manager = TaskManager(include_path=include_paths + [tmpdir])

        print(f"\nRuns: {[r['bench_name'] for r in runs]}")
        print(f"Models: {list(models.keys())}")
        print(f"Device: {device}")
        print("=" * 80)

        all_results = []

        for model_name, model_cfg in models.items():
            print(f"\n{'=' * 80}")
            print(f"Model: {model_name} ({model_cfg['type']})")
            print(f"  args: {build_model_args(model_name, model_cfg)}")
            print(f"{'=' * 80}")

            for run in runs:
                bench_name = run["bench_name"]
                bench_base = run["bench_base"]
                cot_name = run.get("cot_name")

                print(f"\n  [{bench_name}]")
                print(
                    f"    task={run['task']}, fewshot={run['num_fewshot']}, limit={run['limit']}"
                )

                clear_gpu()
                start = time.perf_counter()

                try:
                    eval_kwargs = build_eval_kwargs(
                        model_name, model_cfg, run, task_manager, device
                    )
                    result = simple_evaluate(**eval_kwargs)
                    elapsed = time.perf_counter() - start

                    if result is None:
                        print("    FAILED: returned None")
                        row = {
                            "model": model_name,
                            "benchmark": bench_name,
                            "task": run["task"],
                            "accuracy": None,
                            "error": "returned None",
                            "eval_time_sec": round(elapsed, 2),
                        }
                        all_results.append(row)
                        save_result(results_dir, bench_base, model_name, row, cot_name)
                        continue

                    task_results = result.get("results", {}).get(run["task"], {})
                    accuracy, metric_used = extract_accuracy(task_results)

                    peak_vram_gb = None
                    if torch.cuda.is_available():
                        peak_vram_gb = round(
                            torch.cuda.max_memory_allocated() / 1024**3, 2
                        )

                    row = {
                        "model": model_name,
                        "benchmark": bench_name,
                        "task": run["task"],
                        "accuracy": accuracy,
                        "metric": metric_used if accuracy is not None else None,
                        "num_fewshot": run["num_fewshot"],
                        "limit": run["limit"],
                        "eval_time_sec": round(elapsed, 2),
                        "peak_vram_gb": peak_vram_gb,
                        "error": None,
                    }
                    all_results.append(row)
                    save_result(results_dir, bench_base, model_name, row, cot_name)
                    print(f"    Result: {accuracy} ({metric_used}) in {elapsed:.1f}s")

                except Exception as e:
                    elapsed = time.perf_counter() - start
                    print(f"    FAILED: {e}")
                    row = {
                        "model": model_name,
                        "benchmark": bench_name,
                        "task": run["task"],
                        "accuracy": None,
                        "error": str(e),
                        "eval_time_sec": round(elapsed, 2),
                    }
                    all_results.append(row)
                    save_result(results_dir, bench_base, model_name, row, cot_name)

                clear_gpu()

    df = pd.DataFrame(all_results)

    print(f"\n{'=' * 80}")
    print("Results Summary")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))

    return df
