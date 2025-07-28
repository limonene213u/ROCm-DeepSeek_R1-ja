# R-2 Swallow推論効率測定設定ファイル

# モデル設定
BASELINE_MODEL = "microsoft/DialoGPT-large"  # 32k vocabulary baseline
SWALLOW_MODEL = "tokyotech-llm/Swallow-7b-hf"  # 43k vocabulary Swallow

# 代替モデル（利用可能な場合）
ALTERNATIVE_MODELS = {
    "baseline": [
        "microsoft/DialoGPT-large",
        "rinna/japanese-gpt-1b",
        "cyberagent/open-calm-large"
    ],
    "swallow": [
        "tokyotech-llm/Swallow-7b-hf",
        "tokyotech-llm/Swallow-13b-hf"
    ]
}

# ベンチマーク設定
BENCHMARK_CONFIG = {
    "max_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "n_trials": 3,
    "warmup_runs": 2,
    "confidence_level": 0.95
}

# MI300X最適化設定
HARDWARE_CONFIG = {
    "gpu_memory_utilization": 0.9,
    "enable_chunked_prefill": True,
    "tensor_parallel_size": 1,
    "max_model_len": 2048,
    "dtype": "bfloat16"
}

# 検証目標値
VALIDATION_TARGETS = {
    "speedup_ratio": 1.70,  # 70%高速化目標
    "speedup_percentage": 70.0,
    "paper_claim": "78% inference speedup"
}

# 出力設定
OUTPUT_CONFIG = {
    "results_dir": "results",
    "log_level": "INFO",
    "save_detailed_logs": True,
    "include_system_info": True
}

# プロンプトファイル設定
PROMPT_CONFIG = {
    "default_file": "dataset/prompts_swallow_bench.jsonl",
    "expected_prompts": 30,
    "min_prompt_length": 20,
    "max_prompt_length": 500
}
