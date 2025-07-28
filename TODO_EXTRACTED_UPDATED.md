# TODO/Copilot Instructions Extracted (2025-07-29)

This list consolidates outstanding implementation items after reviewing the codebase and comparing with the research plan.

## High Priority
- Implement Japanese performance benchmarks (R-3/R-4) in `paper_validation_runner.py`.
- Add statistical validation methods (R-7/R-8) using `deepseek_r1_statistical_analysis.R`.
- Provide automation scripts described in the research plan PDF (`environment_setup.py`, `model_downloader.py`, `evaluation_runner.py`, `main.py`).

## Medium Priority
- Implement hipBLASLt performance benchmark referenced in Draft-en.md.
- Complete Rakuten AI 2.0 efficiency evaluation in `paper_validation_suite.py`.
- Expand JLCE 16-task evaluation framework.

## Low Priority
- Remove remaining TODO comments from documentation once validations are implemented.
- Finish the empty R script `analyze_deeepseekr1.r` or remove if unnecessary.

## Current Status
- `mla_kv_cache_benchmark.py` and `lora_efficiency_benchmark.py` execute on CPU or GPU; ROCm required only for GPU acceleration.
- Draft-en.md contains unresolved TODO tags for several claims; these require empirical validation.

