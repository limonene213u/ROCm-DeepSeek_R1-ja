#!/usr/bin/env python3
"""
MLA KVキャッシュ効率測定ベンチマーク

論文記載値「5-13%削減」の実証実験
DeepSeek R1のMulti-Head Latent Attention (MLA) vs 標準Attention

# TODO: Implement baseline attention model comparison
# Copilot: Current implementation only measures DeepSeek MLA
# Need to add Llama-2 or similar model as baseline for "Standard Attention"
# Compare KV cache sizes to validate paper claim "5-13% reduction"
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

@dataclass
class MLABenchmarkConfig:
    """MLA効率測定設定"""
    model_name: str
    sequence_lengths: List[int]
    batch_sizes: List[int]
    precision_modes: List[str]
    num_runs: int
    warmup_runs: int
    output_dir: str

@dataclass
class AttentionBenchmarkResult:
    """Attention効率測定結果"""
    model_name: str
    attention_type: str
    sequence_length: int
    batch_size: int
    precision: str
    kv_cache_memory_mb: float
    attention_computation_time_ms: float
    peak_memory_usage_mb: float
    throughput_tokens_per_sec: float
    measurement_timestamp: str

class MLAEfficiencyMeasurer:
    """MLA効率測定システム"""
    
    def __init__(self, config: MLABenchmarkConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.results: List[AttentionBenchmarkResult] = []
        
        # GPU/ROCm環境設定
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.logger.info(f"GPU Device: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger("MLA_Benchmark")
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラ
        log_file = Path(self.config.output_dir) / "mla_benchmark.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラ
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマッタ
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_model_with_precision(self, precision: str) -> Tuple[Any, Any]:
        """指定精度でモデル読み込み"""
        self.logger.info(f"Loading model {self.config.model_name} with precision {precision}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # 精度設定
        torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def generate_test_sequences(self, tokenizer: Any, seq_length: int, batch_size: int) -> torch.Tensor:
        """テストシーケンス生成"""
        japanese_texts = [
            "日本語の自然言語処理技術は近年大きく進歩しており、特に大規模言語モデルの発展により",
            "機械学習と深層学習の技術革新により、人工知能システムの性能が飛躍的に向上している",
            "トークナイゼーションは自然言語処理において重要な前処理ステップであり、日本語の場合",
            "アテンション機構は現代の言語モデルの中核技術であり、文脈理解能力を大幅に向上させる",
            "ROCmプラットフォームとAMD GPUを活用することで、機械学習の計算効率を最適化できる"
        ]
        
        # バッチサイズ分のテキスト準備
        texts = (japanese_texts * ((batch_size // len(japanese_texts)) + 1))[:batch_size]
        
        # トークナイズ（指定長さに調整）
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=seq_length,
            return_tensors="pt"
        )
        
        return encoded['input_ids'].to(self.device)
    
    def measure_kv_cache_memory(self, model: Any, input_ids: torch.Tensor) -> float:
        """KVキャッシュメモリ使用量測定"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # メモリ使用量測定開始
        initial_memory = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        with torch.no_grad():
            # KVキャッシュ使用での推論
            _ = model(input_ids, use_cache=True)
            
        # ピークメモリ使用量
        peak_memory = torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
        kv_cache_memory = (peak_memory - initial_memory) / (1024 * 1024)  # MB
        
        return kv_cache_memory
    
    def measure_attention_computation_time(self, model: Any, input_ids: torch.Tensor) -> float:
        """Attention計算時間測定"""
        model.eval()
        
        # ウォームアップ
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(input_ids)
        
        # 計算時間測定
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        for _ in range(self.config.num_runs):
            with torch.no_grad():
                _ = model(input_ids)
                
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / self.config.num_runs) * 1000
        return avg_time_ms
    
    def measure_throughput(self, model: Any, tokenizer: Any, input_ids: torch.Tensor) -> float:
        """スループット測定（tokens/sec）"""
        model.eval()
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # 生成実行
            generated = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        # 生成トークン数計算
        new_tokens = generated.shape[1] - input_ids.shape[1]
        total_tokens = new_tokens * input_ids.shape[0]
        
        throughput = total_tokens / (end_time - start_time)
        return throughput
    
    def run_benchmark_single_config(self, seq_length: int, batch_size: int, precision: str) -> AttentionBenchmarkResult:
        """単一設定でのベンチマーク実行"""
        self.logger.info(f"Running benchmark: seq_len={seq_length}, batch_size={batch_size}, precision={precision}")
        
        # モデル読み込み
        model, tokenizer = self.load_model_with_precision(precision)
        
        # テストデータ生成
        input_ids = self.generate_test_sequences(tokenizer, seq_length, batch_size)
        
        # 各種測定
        kv_cache_memory = self.measure_kv_cache_memory(model, input_ids)
        attention_time = self.measure_attention_computation_time(model, input_ids)
        throughput = self.measure_throughput(model, tokenizer, input_ids)
        
        # ピークメモリ使用量
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if self.device == "cuda" else 0
        
        # 結果作成
        result = AttentionBenchmarkResult(
            model_name=self.config.model_name,
            attention_type="MLA" if "deepseek" in self.config.model_name.lower() else "Standard",
            sequence_length=seq_length,
            batch_size=batch_size,
            precision=precision,
            kv_cache_memory_mb=kv_cache_memory,
            attention_computation_time_ms=attention_time,
            peak_memory_usage_mb=peak_memory,
            throughput_tokens_per_sec=throughput,
            measurement_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # メモリクリーンアップ
        del model, tokenizer
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return result
    
    def run_full_benchmark(self) -> List[AttentionBenchmarkResult]:
        """全設定でのベンチマーク実行"""
        self.logger.info("Starting full MLA efficiency benchmark")
        
        results = []
        
        for precision in self.config.precision_modes:
            for seq_length in self.config.sequence_lengths:
                for batch_size in self.config.batch_sizes:
                    try:
                        result = self.run_benchmark_single_config(seq_length, batch_size, precision)
                        results.append(result)
                        self.results.append(result)
                        
                        self.logger.info(f"Completed: {result.attention_type} - KV Cache: {result.kv_cache_memory_mb:.2f}MB")
                        
                    except Exception as e:
                        self.logger.error(f"Failed benchmark for seq_len={seq_length}, batch_size={batch_size}: {e}")
                        continue
        
        return results
    
    def calculate_efficiency_metrics(self, deepseek_results: List[AttentionBenchmarkResult], 
                                   baseline_results: List[AttentionBenchmarkResult]) -> Dict[str, Any]:
        """効率メトリクス計算"""
        metrics = {
            'kv_cache_reduction_percent': [],
            'speed_improvement_percent': [],
            'memory_reduction_percent': [],
            'throughput_improvement_percent': []
        }
        
        for ds_result in deepseek_results:
            # 対応するベースライン結果を探索
            baseline_result = None
            for bl_result in baseline_results:
                if (bl_result.sequence_length == ds_result.sequence_length and
                    bl_result.batch_size == ds_result.batch_size and
                    bl_result.precision == ds_result.precision):
                    baseline_result = bl_result
                    break
            
            if baseline_result:
                # KVキャッシュ削減率
                kv_reduction = ((baseline_result.kv_cache_memory_mb - ds_result.kv_cache_memory_mb) / 
                               baseline_result.kv_cache_memory_mb) * 100
                metrics['kv_cache_reduction_percent'].append(kv_reduction)
                
                # 速度向上
                speed_improvement = ((baseline_result.attention_computation_time_ms - ds_result.attention_computation_time_ms) / 
                                   baseline_result.attention_computation_time_ms) * 100
                metrics['speed_improvement_percent'].append(speed_improvement)
                
                # メモリ削減
                memory_reduction = ((baseline_result.peak_memory_usage_mb - ds_result.peak_memory_usage_mb) / 
                                  baseline_result.peak_memory_usage_mb) * 100
                metrics['memory_reduction_percent'].append(memory_reduction)
                
                # スループット向上
                throughput_improvement = ((ds_result.throughput_tokens_per_sec - baseline_result.throughput_tokens_per_sec) / 
                                        baseline_result.throughput_tokens_per_sec) * 100
                metrics['throughput_improvement_percent'].append(throughput_improvement)
        
        # 統計計算
        summary_metrics = {}
        for key, values in metrics.items():
            if values:
                summary_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        return summary_metrics
    
    def save_results(self, filename: str = None):
        """結果保存"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"mla_benchmark_results_{timestamp}.json"
        
        output_path = Path(self.config.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 結果をJSON形式で保存
        results_dict = {
            'config': {
                'model_name': self.config.model_name,
                'sequence_lengths': self.config.sequence_lengths,
                'batch_sizes': self.config.batch_sizes,
                'precision_modes': self.config.precision_modes,
                'num_runs': self.config.num_runs
            },
            'results': [
                {
                    'model_name': r.model_name,
                    'attention_type': r.attention_type,
                    'sequence_length': r.sequence_length,
                    'batch_size': r.batch_size,
                    'precision': r.precision,
                    'kv_cache_memory_mb': r.kv_cache_memory_mb,
                    'attention_computation_time_ms': r.attention_computation_time_ms,
                    'peak_memory_usage_mb': r.peak_memory_usage_mb,
                    'throughput_tokens_per_sec': r.throughput_tokens_per_sec,
                    'measurement_timestamp': r.measurement_timestamp
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
        return output_path

def main():
    """メイン実行関数"""
    # 設定
    config = MLABenchmarkConfig(
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        sequence_lengths=[512, 1024, 2048, 4096],
        batch_sizes=[1, 2, 4, 8],
        precision_modes=["fp16", "bf16"],
        num_runs=5,
        warmup_runs=2,
        output_dir="./benchmark_results"
    )
    
    # ベンチマーク実行
    measurer = MLAEfficiencyMeasurer(config)
    results = measurer.run_full_benchmark()
    
    # 結果保存
    measurer.save_results()
    
    # 簡易レポート出力
    print(f"\nMLA Efficiency Benchmark Complete")
    print(f"Total measurements: {len(results)}")
    print(f"Model: {config.model_name}")
    
    if results:
        avg_kv_memory = np.mean([r.kv_cache_memory_mb for r in results])
        avg_throughput = np.mean([r.throughput_tokens_per_sec for r in results])
        print(f"Average KV Cache Memory: {avg_kv_memory:.2f} MB")
        print(f"Average Throughput: {avg_throughput:.2f} tokens/sec")

if __name__ == "__main__":
    main()
