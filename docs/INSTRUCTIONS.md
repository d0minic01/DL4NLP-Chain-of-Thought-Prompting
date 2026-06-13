# Running the Project

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Python | 3.13 | [python.org](https://python.org) or `winget install Python.Python.3.13` |
| uv | any recent | `pip install uv` or `winget install astral-sh.uv` |
| git | any | already installed if you cloned this repo |

A GPU is **not required**. All models listed for laptop testing run fine on CPU, just slower.

---

## 1. Install dependencies

```bash
uv sync
```

This creates `.venv/` and installs everything from `uv.lock` (lm-eval, transformers, torch, matplotlib, etc.).

---

## 2. Quick smoke test (2 minutes on CPU)

Run one tiny model on one benchmark with 5 examples to confirm the pipeline works end-to-end:

```bash
# Baseline (standard few-shot, no chain-of-thought)
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 5

# CoT variant of the same
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 5 -c
```

Results are saved to `results/gsm8k/HuggingFaceTB_SmolLM2-135M-Instruct/`.

---

## 3. Laptop-appropriate models

These fit in RAM on a typical laptop CPU (no GPU needed). Pick one to start:

| Model | Params | RAM needed | Speed (CPU) |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135 M | ~0.5 GB | fast |
| `google/flan-t5-small` | 80 M | ~0.5 GB | fast |
| `google/gemma-3-270m-it` | 270 M | ~1 GB | fast |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360 M | ~1.5 GB | moderate |
| `Qwen/Qwen2.5-0.5B-Instruct` | 500 M | ~2 GB | moderate |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7 B | ~4 GB | slow on CPU |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | ~4 GB | slow on CPU |

For a laptop with 8 GB RAM and no GPU, stick to the first four. The 1.5 B+ models will be slow on CPU but will run.

---

## 4. Running specific scenarios

All commands follow this pattern:

```
uv run python run_eval.py [options]
```

### Flags

| Flag | What it does |
|---|---|
| `-m MODEL` | Run only this model (HuggingFace ID, must be in `benchmarks.yaml`) |
| `-b BENCHMARK` | Run only this benchmark (name from `benchmarks.yaml`) |
| `-c` | Use CoT prompts instead of baseline few-shot |
| `-p PROMPT_SET` | Override the prompt set (file name in `fewshots_and_cots/`, without `.jsonl`) |
| `-l N` | Limit to N examples (use 20–50 on a laptop for speed) |
| `-d` | Debug: save full model inputs/outputs alongside CSVs |
| `-v` | Print each sample's prompt, model output, and extracted answer to stdout |
| `--output-file FILE` | Write verbose output to FILE instead of stdout (implies `-v`) |

### Common recipes

```bash
# One model, one benchmark, both variants (baseline then CoT)
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 20
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 20 -c

# Try a symbolic benchmark (coin flip)
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b coin_flip -l 20
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b coin_flip -l 20 -c

# Try an extended prompting technique on GSM8K
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 20 -p equation_only
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 20 -p numbered_step_cot
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 20 -p contrastive_cot

# Inspect prompts and outputs live in the terminal
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 5 -v

# Same but write to a file (useful for longer runs or sharing)
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 20 --output-file gsm8k_log.txt

# Combine with CoT to see the reasoning chains the model produces
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 5 -c -v

# Save raw JSON for later analysis (separate from -v)
uv run python run_eval.py -m HuggingFaceTB/SmolLM2-135M-Instruct -b gsm8k -l 5 -c -d
```

### Available benchmarks

| Name | Category | Description |
|---|---|---|
| `gsm8k` | arithmetic | Grade-school math word problems |
| `asdiv` | arithmetic | Diverse arithmetic word problems |
| `mawps` | arithmetic | Math word problem benchmark |
| `svamp` | arithmetic | Adversarial math word problems |
| `aqua` | arithmetic | Algebraic problems with multiple-choice answers |
| `csqa` | commonsense | CommonsenseQA multiple-choice |
| `strategyqa` | commonsense | Yes/no questions requiring multi-hop reasoning |
| `sports` | commonsense | Athlete–action plausibility (yes/no) |
| `date` | commonsense | Date arithmetic (MM/DD/YYYY output) |
| `coin_flip` | symbolic | Track coin state through a flip sequence |
| `last_letter` | symbolic | Concatenate last letters of words in a name |
| `letter_shift` | symbolic | Track letter position through shift operations |

### Available prompt sets (`-p`)

| Name | Description |
|---|---|
| `math_word_problems` | Original paper few-shot CoT (arithmetic) |
| `math_word_problems_aqua` | Original paper few-shot CoT (AQuA multiple-choice) |
| `csqa` | Original paper few-shot CoT (CommonsenseQA) |
| `strategy_qa` | Original paper few-shot CoT (StrategyQA) |
| `sports_understanding` | Original paper few-shot CoT (sports plausibility) |
| `date_understanding` | Original paper few-shot CoT (date arithmetic) |
| `coin_flip` | Original paper few-shot CoT (coin flip) |
| `last_letter_concatenation` | Original paper few-shot CoT (letter concat) |
| `letter_shift` | Custom benchmark few-shot CoT |
| `equation_only` | Ablation: reasoning = bare equation, no language |
| `numbered_step_cot` | Variant: explicit Step 1 / Step 2 / … numbering |
| `persona_cot` | Variant: "As a mathematician: …" role framing |
| `contrastive_cot` | Variant: [INCORRECT] + [CORRECT] chain per exemplar |

---

## 5. Understanding results

Each run creates a CSV in `results/<benchmark>/<model>/`:

```
results/
  gsm8k/
    HuggingFaceTB_SmolLM2-135M-Instruct/
      HuggingFaceTB_SmolLM2-135M-Instruct.csv          ← baseline run
      HuggingFaceTB_SmolLM2-135M-Instruct-math_word_problems.csv  ← CoT run
```

Each CSV has one row with columns: `model`, `benchmark`, `accuracy`, `metric`, `num_fewshot`, `limit`, `eval_time_sec`, `peak_vram_gb`.

The key comparison is `accuracy` between the baseline CSV and the CoT CSV for the same model and benchmark.

---

## 6. Visualizing results

After collecting some results, generate plots:

```bash
uv run python visualize.py           # saves to plots/
uv run python visualize.py -o my_plots  # custom output folder
```

Four plots are produced:

| File | Description |
|---|---|
| `<model>_benchmark_comparison.png` | Baseline vs CoT bars for each benchmark |
| `cot_delta_heatmap.png` | CoT gain matrix (model × benchmark) |
| `scaling_curves.png` | Accuracy vs model size (log scale), per task category |
| `category_summary.png` | Average CoT gain across arithmetic / commonsense / symbolic |

---

## 7. Regenerating test data (optional)

The `data/` files are already committed. Only run this if you modify the generator or want to re-seed:

```bash
uv run python scripts/generate_symbolic_data.py   # coin_flip, last_letter, date, sports
uv run python scripts/generate_letter_shift_data.py  # letter_shift
```

---

## 8. Running the full experiment (GTX 1080 Ti)

Once the pipeline is validated on a laptop, run all models and benchmarks on the workstation:

```bash
# Full run — all models, all benchmarks, baseline + CoT
uv run python run_eval.py

# Or selectively by category
uv run python run_eval.py -b gsm8k    # all models, one benchmark, baseline + CoT
uv run python run_eval.py -b gsm8k -c # CoT only
```

Expected runtimes per model (rough estimates, 1080 Ti, full dataset):
- 135 M – 270 M models: ~5–15 min per benchmark
- 1 B – 2 B models: ~20–40 min per benchmark
- 7 B models: ~2–4 hours per benchmark

Run benchmarks with limited examples first to catch any model-specific issues:

```bash
uv run python run_eval.py -l 50
```

Models that may need special attention:
- **Flan-T5** (encoder-decoder): watch for unexpected output format — lm-eval handles seq2seq but the few-shot prompt structure differs from decoder-only models.
- **Gemma-2-9B / Qwen2.5-7B**: may require reducing `batch_size` to 1 in `benchmarks.yaml` if VRAM is exceeded.
- **DeepSeek-R1-Distill**: these models produce long reasoning traces before the answer; the flexible-extract filter handles this but strict-match accuracy will be lower.
