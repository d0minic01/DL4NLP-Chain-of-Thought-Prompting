import gc
import time

import pandas as pd
import torch
from lm_eval import simple_evaluate

MODELS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    # "t5-11b",
]

TASK = "gsm8k"
LIMIT = 100


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


results = []

for model_name in MODELS:
    clear_gpu()

    start_total = time.perf_counter()

    try:
        start_eval = time.perf_counter()

        result = simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name}",
            tasks=[TASK],
            batch_size="auto",
            device="cuda",
            limit=LIMIT,
            log_samples=False,
        )

        eval_time = time.perf_counter() - start_eval
        total_time = time.perf_counter() - start_total

        accuracy = result["results"].get(TASK, {}).get("exact_match,none", None)

        peak_vram_gb = None

        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "eval_time_sec": round(eval_time, 2),
                "total_time_sec": round(total_time, 2),
                "samples_per_sec": round(LIMIT / eval_time, 3),
                "peak_vram_gb": (round(peak_vram_gb, 2) if peak_vram_gb else None),
            }
        )

    except Exception as e:
        results.append({"model": model_name, "error": str(e)})

    clear_gpu()


df = pd.DataFrame(results)

print(df)

df.to_csv("t5_benchmark_results.csv", index=False)

print("Saved t5_benchmark_results.csv")
