# Chain-of-Thought Prompting Evaluation

## Setup

```bash
uv sync
```



## Flags

| Flag | Description |
|---|---|
| `-m, --model` | Model name defined in `benchmarks.yaml` |
| `-b, --benchmark` | Benchmark name defined in `benchmarks.yaml` |
| `-c, --cot` | Run the CoT variant (default: runs the baseline fewshot variant) |
| `-p, --prompts PROMPT_SET` | Override the prompt set (default: uses the one defined in `benchmarks.yaml`) |
| `-l, --limit N` | Override the number of examples per benchmark |
| `-d, --debug` | Save model inputs/outputs to JSON alongside results |

## Behavior

| Command | Runs |
|---|---|
| `./run_eval.py` | All benchmarks × all models (both variants) |
| `./run_eval.py -m model` | All benchmarks for one model (both variants) |
| `./run_eval.py -b bench` | One benchmark for all models (both variants) |
| `./run_eval.py -m model -b bench` | Baseline fewshot only |
| `./run_eval.py -m model -b bench -c` | CoT variant only |
| `./run_eval.py -m model -b bench -c -p prompts` | CoT with overridden prompt set |

## Adding prompt sets

Drop a `.jsonl` file into `fewshots_and_cots/`. Each line is one example:

```json
{"question": "If there are 3 cars and 2 more arrive, how many total?", "chain-of-thought": "3 + 2 = 5.", "target": "5"}
```
