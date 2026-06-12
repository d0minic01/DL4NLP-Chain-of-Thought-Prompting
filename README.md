# Chain-of-Thought Prompting Evaluation

## Setup

```bash
uv sync
```

## Flags

| Flag | Description |
|---|---|
| `-m, --model` | Model name defined in `benchmarks.yaml` (omit to run all models) |
| `-b, --benchmark` | Benchmark name defined in `benchmarks.yaml` (omit to run all benchmarks) |
| `-c, --cot` | Run the CoT variant instead of the baseline fewshot variant |
| `-p, --prompts PROMPT_SET` | Override which prompt set to use (independent of `-c`) |
| `-l, --limit N` | Override the number of examples per benchmark |
| `-d, --debug` | Save model inputs/outputs to JSON alongside results |

## Examples

```bash
# Run everything
./run_eval.py

# One model, all benchmarks
./run_eval.py -m Qwen/Qwen3-0.6B

# One benchmark, all models — baseline fewshot
./run_eval.py -b gsm8k

# One model, one benchmark — baseline fewshot
./run_eval.py -m Qwen/Qwen3-0.6B -b gsm8k

# CoT variant
./run_eval.py -m Qwen/Qwen3-0.6B -b gsm8k -c

# Override the prompt set (works with or without -c)
./run_eval.py -m Qwen/Qwen3-0.6B -b gsm8k -p math_word_problems
./run_eval.py -m Qwen/Qwen3-0.6B -b gsm8k -c -p math_word_problems

# Limit examples and save outputs for inspection
./run_eval.py -m Qwen/Qwen3-0.6B -b gsm8k -l 50 -d
```

## Adding prompt sets

Drop a `.jsonl` file into `fewshots_and_cots/`. Each line is one example:

```json
{"question": "If there are 3 cars and 2 more arrive, how many total?", "chain-of-thought": "3 + 2 = 5.", "target": "5"}
```
