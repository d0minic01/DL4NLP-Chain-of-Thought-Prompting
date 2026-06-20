import gc
import json
import re
import sys
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
    discover_fewshot_sets,
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
    model_args = {"pretrained": model_name}
    extra = model_cfg.get("model_args", {})
    if extra:
        model_args.update(extra)

    kwargs = dict(
        model="hf",
        model_args=model_args,
        tasks=[run["bench_name"]],
        log_samples=log_samples,
        task_manager=task_manager,
        device=device,
        apply_chat_template=model_cfg.get("apply_chat_template", False),
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


def save_result(results_dir, bench_base, model_name, row, technique=None, samples=None):

    safe_name = model_name.replace("/", "_").replace(":", "_")
    model_dir = results_dir / bench_base / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{safe_name}-{technique}.csv" if technique else f"{safe_name}.csv"
    csv_path = model_dir / csv_filename
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    if samples is not None:
        json_filename = (
            f"{safe_name}-{technique}-samples.json"
            if technique
            else f"{safe_name}-samples.json"
        )
        json_path = model_dir / json_filename
        with open(json_path, "w") as f:
            json.dump(samples, f, indent=2, default=str)


def _get_extracted_answer(sample: dict) -> str:
    fr = sample.get("filtered_resps")
    if isinstance(fr, dict):
        for key in ("strict-match", next(iter(fr), None)):
            if key and fr.get(key):
                val = fr[key]
                return val[0] if isinstance(val, list) else str(val)
    elif isinstance(fr, list) and fr:
        first = fr[0]
        return first[0] if isinstance(first, list) and first else str(first)
    return ""


def _extract_test_question(prompt: str) -> str:
    """Return only the final test question from a prompt, stripping the few-shot context."""
    parts = prompt.rsplit("\n\nQuestion:", 1)
    if len(parts) == 2:
        return "Question:" + parts[1]
    return prompt


def print_sample(sample: dict, bench_base: str, technique: str, idx: int, total: int, out, num_correct: int = 0) -> bool | None:
    prompt = ""
    if sample.get("arguments"):
        args0 = sample["arguments"][0]
        prompt = args0[0] if args0 else ""

    raw = ""
    if sample.get("resps"):
        resps0 = sample["resps"][0]
        raw = resps0[0] if resps0 else ""

    extracted = _get_extracted_answer(sample)
    target = str(sample.get("target", ""))

    # Try metric keys first; fall back to direct string comparison
    correct = None
    for key in ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc,none"):
        if key in sample:
            correct = bool(sample[key])
            break
    if correct is None and extracted:
        correct = extracted.strip().lower() == target.strip().lower()

    verdict = (" CORRECT" if correct else " WRONG") if correct is not None else ""
    running_correct = num_correct + (1 if correct else 0)
    score_tag = f"  |  {running_correct}/{idx + 1} correct" if correct is not None else ""

    SEP  = "=" * 64
    LINE = "-" * 64

    question_text = re.sub(r"\s*\n?\s*Answer:\s*$", "", _extract_test_question(prompt)).rstrip()

    raw_stripped = raw.strip() if raw else ""
    match = re.search(r"(The answer is\b.*)", raw_stripped, re.IGNORECASE | re.DOTALL) if raw_stripped else None
    reasoning = raw_stripped[:match.start()].rstrip() if match else raw_stripped or "(empty)"
    final_answer = match.group(1).strip() if match else None

    print(SEP, file=out)
    print(f"  {bench_base}  |  {technique}  |  sample {idx + 1}/{total}{score_tag}  |  {verdict}", file=out)
    print(SEP, file=out)
    print(question_text, file=out)
    if reasoning:
        print(f"Model Reasoning: {reasoning}", file=out)
    if final_answer:
        print(f"Final Answer: {final_answer}", file=out)
    if not reasoning and not final_answer:
        print(f"Model Output: {raw_stripped or '(empty)'}", file=out)
    print(LINE, file=out)
    print(f"  extracted: {extracted!r:<10}  target: {target!r}", file=out)
    print(file=out)
    return correct


def run_eval(config):
    benchmarks_cfg = config.get("benchmarks", {})
    cot_techniques_cfg = config.get("cot_techniques", {})
    cot_files = discover_fewshot_sets()
    models = resolve_models(config.get("models", {}))
    results_dir = Path("results")

    runs = configure_runs(benchmarks_cfg, cot_techniques_cfg, cot_files)

    if technique_override := config.get("technique_override"):
        runs = [r for r in runs if r["technique"] in technique_override]

    if config.get("limit") is not None:
        for run in runs:
            run["limit"] = config["limit"]

    verbose = config.get("verbose", False)
    if verbose:
        config["log_samples"] = True

    # read after verbose may have forced it on
    log_samples = config.get("log_samples", False)

    output_file_path = config.get("output_file")
    out_stream = open(output_file_path, "w", encoding="utf-8") if output_file_path else None

    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdir:
        for run in runs:
            yaml_content, num_fewshot = build_task_yaml(
                run["task"],
                run["bench_name"],
                run["technique_type"],
                run["questions_path"],
                technique_path=run.get("technique_path"),
                instruction=run.get("instruction"),
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
                technique = run["technique"]

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

                    if verbose and samples:
                        out = out_stream if out_stream else sys.stdout
                        # lm_eval emits one entry per (doc, filter_key) pair;
                        # keep only the first entry per doc to avoid duplicates.
                        seen_ids: set = set()
                        display_samples = []
                        for s in samples:
                            doc_id = s.get("doc_id")
                            if doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                display_samples.append(s)
                        num_correct = 0
                        for i, sample in enumerate(display_samples):
                            result_correct = print_sample(sample, bench_base, technique, i, len(display_samples), out, num_correct)
                            if result_correct:
                                num_correct += 1
                        if out_stream:
                            out_stream.flush()

                except Exception as e:
                    elapsed = time.perf_counter() - start
                    accuracy = None
                    metric_used = None
                    peak_vram_gb = None
                    print(f"    FAILED: {e}")

                row = {
                    "model": model_name,
                    "benchmark": bench_name,
                    "bench_base": bench_base,
                    "technique": technique,
                    "technique_type": run["technique_type"],
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
                    technique,
                    samples=samples,
                )

    clear_gpu()

    if out_stream:
        out_stream.close()

    df = pd.DataFrame(all_results)

    return df
