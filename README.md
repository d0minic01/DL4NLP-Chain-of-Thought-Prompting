# Chain-of-Thought Prompting Evaluation

Benchmark evaluation framework for comparing models with and without chain-of-thought prompting, built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Setup

```bash
uv sync
```

For Ollama models, ensure Ollama is running locally (`ollama serve`). For API, ensure the respective API key is set as an environment variable.

## Quick Start

```bash
# Run all benchmarks × all cots × all models
python run_eval.py

# Single benchmark, single model
python run_eval.py -m gemma4:e4b -b gsm8k

# Single benchmark with a specific CoT variant
python run_eval.py -m gemma4:e4b -b gsm8k -c example-1
```

When using `-b` or `-c`, `-m` is required. Bare `python run_eval.py` runs everything.

## CLI Flags

| Flag | Description |
|---|---|
| `-m, --model` | Model name from `benchmarks.yaml` |
| `-b, --benchmark` | Benchmark name from `benchmarks.yaml` |
| `-c, --cot` | CoT variant name from `benchmarks.yaml` |

### Behavior

| Flags | Runs |
|---|---|
| _(none)_ | All benchmarks × all cots × all models |
| `-m model` | All benchmarks × all cots × one model |
| `-m model -b bench` | Base benchmark only (no cots) × one model |
| `-m model -b bench -c cot` | Single cot variant only × one model |
| `-m model -c cot` | All benchmarks × one cot × one model |

## Configuration (`benchmarks.yaml`)

### Benchmarks

Each benchmark references an lm_eval task. Every enabled benchmark runs with every CoT variant by default.

```yaml
benchmarks:
  gsm8k:
    task: gsm8k        # lm_eval task name
    enabled: true
    num_fewshot: 5
    limit: null         # null = all samples, integer = cap
```

`cot_overrides` are needed when the base task type doesn't support generation (e.g. `loglikelihood` → `generate_until` for asdiv, `multiple_choice` → `generate_until` for csqa).

### CoT Variants

CoT configs define few-shot examples. They're applied to all benchmarks automatically. Task-specific prompt/filter overrides go in `benchmarks.cot_overrides`, not here.

```yaml
cots:
  example-1:
    fewshot_path: fewshot/example-1.jsonl
    num_fewshot: 8
```

Few-shot JSONL format — one JSON object per line with `question` and `target` fields:
```json
{"question": "If there are 3 cars and 2 more arrive, how many total?", "target": "3 + 2 = 5. The final answer is 5"}
```

### Models

Models are defined inline. All API types use `model=<name>` with optional `base_url`. HuggingFace uses `pretrained=<name>`.

```yaml
models:
  defaults:
    apply_chat_template: true
    fewshot_as_multiturn: false
    batch_size: null

  # Ollama
  gemma4:e4b:
    type: local-chat-completions
    base_url: "http://localhost:11434/v1/chat/completions"

  # HuggingFace
  t5-small:
    type: hf
    batch_size: "auto"

  # OpenAI (needs OPENAI_API_KEY)
  gpt-4o:
    type: openai-chat-completions

  # Anthropic (needs ANTHROPIC_API_KEY)
  claude-3-opus:
    type: anthropic-chat

  # OpenRouter (needs OPENROUTER_API_KEY)
  llama-3.1-8b:
    type: openai-chat-completions
    base_url: "https://openrouter.ai/api/v1"
```

## Output Structure

```
results/
  <benchmark>/
    <model>/
      <model>.csv              # base run
      <model>-<cot>.csv        # cot variant run
```

## Project Structure

```
run_eval.py          # Entry point
benchmarks.yaml      # Configuration
fewshot/             # CoT few-shot example files
tasks/               # Custom lm_eval task definitions
eval/
  cli.py             # Argument parsing, config filtering
  tasks.py           # Config loading, run expansion, CoT task generation
  runner.py          # Evaluation loop, result saving
```