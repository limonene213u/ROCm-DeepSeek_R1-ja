# SwallowInferenceBenchmark Implementation Overview

## File: `Python/Benchmark/swallow_inference_benchmark.py`

This script measures inference throughput of the Swallow continual pre-training models against a baseline model. It is designed for ROCm environments and records tokens-per-second, latency, and memory consumption.

### Vocabulary Compensation
To compare a 32k vocabulary baseline with the 43k-token Swallow model, the benchmark now adjusts the baseline throughput by `baseline_vocab / swallow_vocab`. The resulting adjusted tokens-per-second values are stored alongside the raw measurements.

```python
baseline_adj_tps = baseline_result.tokens_per_sec * (baseline_vocab / swallow_vocab)
swallow_adj_tps = swallow_result.tokens_per_sec
speedup_ratio = swallow_adj_tps / baseline_adj_tps
```

### CLI Usage
The script can be run directly:
```bash
python Python/Benchmark/swallow_inference_benchmark.py \
    --baseline microsoft/DialoGPT-large \
    --swallow tokyotech-llm/Swallow-7b-hf \
    --prompts dataset/prompts_swallow_bench.jsonl
```
The output JSON includes adjusted throughput metrics and a PASS/FAIL indicator relative to the 70% speedup target.
