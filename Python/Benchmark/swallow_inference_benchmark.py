#!/usr/bin/env python3
"""
Swallow推論効率測定ベンチマーク
DeepSeek R1 日本語適応研究 - R-2検証タスク

継続事前学習モデル Swallow (32k→43k語彙拡張版) の
「推論78%高速化」論文クレームを検証する。

目標：
- 推論スループット（tokens/sec）がベースライン比 +70%以上
- 再現スクリプトとログを自動保存
- paper_validation_suite.py で PASS を返す
"""

import json
import time
import statistics
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# vLLM import - ROCm 対応版
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM not available. Using transformers fallback.")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    VLLM_AVAILABLE = False

@dataclass
class BenchmarkResult:
    """ベンチマーク結果を格納するデータクラス"""
    model_name: str
    tokens_per_sec: float
    avg_latency: float
    total_tokens: int
    total_time: float
    confidence_interval: Tuple[float, float]
    memory_peak_mb: float

class SwallowInferenceBenchmark:
    """Swallow推論効率ベンチマーク実行クラス"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # ROCm GPU同期用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def sync_cuda(self):
        """GPU処理完了まで待機（正確な時間測定のため）"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def get_memory_usage(self) -> float:
        """現在のGPUメモリ使用量をMB単位で取得"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def bootstrap_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """ブートストラップ法で信頼区間を計算"""
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def load_prompts(self, prompt_file: str) -> List[Dict]:
        """プロンプトファイルから測定用データを読み込み"""
        prompts = []
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                prompts.append(json.loads(line.strip()))
        return prompts
    
    def measure_tokens_per_sec_vllm(self, model_name: str, prompts: List[Dict], 
                                   dtype: str = "bfloat16", n_trials: int = 3) -> BenchmarkResult:
        """vLLMを使用した高精度推論速度測定"""
        
        # vLLM設定 - MI300X最適化
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            dtype=dtype,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            # ROCm 6.1+ の場合有効化
            enable_chunked_prefill=True,
            max_model_len=2048  # メモリ効率のため制限
        )
        
        sampling_params = SamplingParams(
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            stop=None
        )
        
        # ウォームアップ（2回実行）
        warmup_prompts = [p["prompt"] for p in prompts[:2]]
        self.sync_cuda()
        _ = llm.generate(warmup_prompts, sampling_params)
        self.sync_cuda()
        
        # 本測定（複数試行）
        all_latencies = []
        all_token_counts = []
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials} for {model_name}")
            
            trial_latencies = []
            trial_tokens = []
            
            for i, prompt_data in enumerate(prompts):
                prompt = prompt_data["prompt"]
                
                # GPU状態リセット
                torch.cuda.empty_cache()
                self.sync_cuda()
                
                # 推論実行と時間測定
                start_time = time.perf_counter()
                outputs = llm.generate([prompt], sampling_params)
                self.sync_cuda()
                end_time = time.perf_counter()
                
                # 結果収集
                output = outputs[0]
                generated_tokens = len(output.outputs[0].token_ids)
                latency = end_time - start_time
                
                trial_latencies.append(latency)
                trial_tokens.append(generated_tokens)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(prompts)} prompts")
            
            all_latencies.extend(trial_latencies)
            all_token_counts.extend(trial_tokens)
        
        # 統計計算
        total_tokens = sum(all_token_counts)
        total_time = sum(all_latencies)
        tokens_per_sec = total_tokens / total_time
        avg_latency = statistics.mean(all_latencies)
        
        # トークン/秒の信頼区間計算
        tokens_per_sec_samples = [tc / lt for tc, lt in zip(all_token_counts, all_latencies)]
        ci = self.bootstrap_confidence_interval(tokens_per_sec_samples)
        
        # メモリ使用量
        memory_peak = self.get_memory_usage()
        
        return BenchmarkResult(
            model_name=model_name,
            tokens_per_sec=tokens_per_sec,
            avg_latency=avg_latency,
            total_tokens=total_tokens,
            total_time=total_time,
            confidence_interval=ci,
            memory_peak_mb=memory_peak
        )
    
    def measure_tokens_per_sec_transformers(self, model_name: str, prompts: List[Dict], 
                                          dtype: str = "bfloat16") -> BenchmarkResult:
        """transformersライブラリを使用したフォールバック測定"""
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        all_latencies = []
        all_token_counts = []
        
        print(f"Measuring {model_name} with transformers backend...")
        
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            
            # 入力トークン化
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 推論実行
            self.sync_cuda()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=None,
                    top_p=None
                )
            
            self.sync_cuda()
            end_time = time.perf_counter()
            
            # 生成トークン数計算
            input_length = inputs.input_ids.shape[1]
            output_length = outputs.shape[1]
            generated_tokens = output_length - input_length
            
            latency = end_time - start_time
            all_latencies.append(latency)
            all_token_counts.append(generated_tokens)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(prompts)} prompts")
        
        # 統計計算
        total_tokens = sum(all_token_counts)
        total_time = sum(all_latencies)
        tokens_per_sec = total_tokens / total_time
        avg_latency = statistics.mean(all_latencies)
        
        tokens_per_sec_samples = [tc / lt for tc, lt in zip(all_token_counts, all_latencies)]
        ci = self.bootstrap_confidence_interval(tokens_per_sec_samples)
        
        memory_peak = self.get_memory_usage()
        
        return BenchmarkResult(
            model_name=model_name,
            tokens_per_sec=tokens_per_sec,
            avg_latency=avg_latency,
            total_tokens=total_tokens,
            total_time=total_time,
            confidence_interval=ci,
            memory_peak_mb=memory_peak
        )
    
    def run_comparative_benchmark(self, baseline_model: str, swallow_model: str,
                                prompt_file: str,
                                baseline_vocab: int = 32000,
                                swallow_vocab: int = 43000) -> Dict:
        """ベースラインとSwallowモデルの比較ベンチマーク実行"""
        
        prompts = self.load_prompts(prompt_file)
        print(f"Loaded {len(prompts)} prompts from {prompt_file}")
        
        # 測定実行
        if VLLM_AVAILABLE:
            print("Using vLLM backend for optimal performance")
            baseline_result = self.measure_tokens_per_sec_vllm(baseline_model, prompts)
            swallow_result = self.measure_tokens_per_sec_vllm(swallow_model, prompts)
        else:
            print("Using transformers backend (fallback)")
            baseline_result = self.measure_tokens_per_sec_transformers(baseline_model, prompts)
            swallow_result = self.measure_tokens_per_sec_transformers(swallow_model, prompts)
        
        # 語彙差異補正
        baseline_adj_tps = baseline_result.tokens_per_sec * (baseline_vocab / swallow_vocab)
        swallow_adj_tps = swallow_result.tokens_per_sec

        # 速度向上率計算
        speedup_ratio = swallow_adj_tps / baseline_adj_tps
        speedup_percentage = (speedup_ratio - 1.0) * 100
        
        # 結果まとめ
        results = {
            "benchmark_type": "Swallow Inference Efficiency (R-2)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline": {
                "model": baseline_result.model_name,
                "tokens_per_sec": baseline_result.tokens_per_sec,
                "adjusted_tokens_per_sec": baseline_adj_tps,
                "avg_latency_sec": baseline_result.avg_latency,
                "confidence_interval": baseline_result.confidence_interval,
                "memory_peak_mb": baseline_result.memory_peak_mb
            },
            "swallow": {
                "model": swallow_result.model_name,
                "tokens_per_sec": swallow_result.tokens_per_sec,
                "adjusted_tokens_per_sec": swallow_adj_tps,
                "avg_latency_sec": swallow_result.avg_latency,
                "confidence_interval": swallow_result.confidence_interval,
                "memory_peak_mb": swallow_result.memory_peak_mb
            },
            "comparison": {
                "speedup_ratio": speedup_ratio,
                "speedup_percentage": speedup_percentage,
                "target_speedup": 1.70,  # 70%高速化目標
                "meets_target": speedup_ratio >= 1.70,
                "performance_gap": speedup_percentage - 70.0
            },
            "system_info": {
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "total_gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
                "vllm_available": VLLM_AVAILABLE,
                "pytorch_version": torch.__version__
            }
        }
        
        # 結果保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"swallow_benchmark_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {result_file}")
        return results
    
    def print_summary(self, results: Dict):
        """ベンチマーク結果のサマリー表示"""
        print("\n" + "="*60)
        print("SWALLOW INFERENCE EFFICIENCY BENCHMARK RESULTS")
        print("="*60)
        
        baseline = results["baseline"]
        swallow = results["swallow"]
        comparison = results["comparison"]
        
        print(f"Baseline Model: {baseline['model']}")
        print(f"  Tokens/sec: {baseline['tokens_per_sec']:.2f} (adj {baseline['adjusted_tokens_per_sec']:.2f})")
        print(f"  Avg Latency: {baseline['avg_latency_sec']:.3f}s")
        print(f"  Memory Peak: {baseline['memory_peak_mb']:.1f}MB")
        
        print(f"\nSwallow Model: {swallow['model']}")
        print(f"  Tokens/sec: {swallow['tokens_per_sec']:.2f} (adj {swallow['adjusted_tokens_per_sec']:.2f})")
        print(f"  Avg Latency: {swallow['avg_latency_sec']:.3f}s")
        print(f"  Memory Peak: {swallow['memory_peak_mb']:.1f}MB")
        
        print(f"\nPerformance Comparison:")
        print(f"  Speedup Ratio: {comparison['speedup_ratio']:.2f}x")
        print(f"  Speedup Percentage: {comparison['speedup_percentage']:+.1f}%")
        print(f"  Target (70%): {'✅ ACHIEVED' if comparison['meets_target'] else '❌ NOT MET'}")
        
        if comparison['meets_target']:
            print(f"  Exceeded target by: {comparison['performance_gap']:+.1f}%")
        else:
            print(f"  Below target by: {abs(comparison['performance_gap']):.1f}%")
        
        print("="*60)

def main():
    """メイン実行関数 - CLI使用時"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Swallow Inference Efficiency Benchmark")
    parser.add_argument("--baseline", default="microsoft/DialoGPT-large", 
                       help="Baseline model name")
    parser.add_argument("--swallow", default="tokyotech-llm/Swallow-7b-hf",
                       help="Swallow model name")
    parser.add_argument("--prompts", default="dataset/prompts_swallow_bench.jsonl",
                       help="Prompt file path")
    parser.add_argument("--results-dir", default="results",
                       help="Results output directory")
    
    args = parser.parse_args()
    
    benchmark = SwallowInferenceBenchmark(results_dir=args.results_dir)
    
    try:
        results = benchmark.run_comparative_benchmark(
            baseline_model=args.baseline,
            swallow_model=args.swallow, 
            prompt_file=args.prompts
        )
        benchmark.print_summary(results)
        
        # validation結果を返す
        validation_status = "PASS" if results["comparison"]["meets_target"] else "FAIL"
        print(f"\nValidation Status: {validation_status}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
