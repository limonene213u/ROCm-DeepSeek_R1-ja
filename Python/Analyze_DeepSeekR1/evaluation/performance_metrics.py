#!/usr/bin/env python3
"""
Performance Metrics Calculator
Statistical analysis and performance measurement utilities

Author: Akira Ito a.k.a limonene213u
統計解析と性能測定のユーティリティスクリプト
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime
import logging

# パス設定の修正
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not available. System monitoring will be limited.")
    PSUTIL_AVAILABLE = False

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """性能測定と統計分析クラス"""
    
    def __init__(self):
        self.metrics_history = []
        
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """関数実行時間の測定"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    def measure_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """メモリ使用量の測定"""
        if not PSUTIL_AVAILABLE:
            result = func(*args, **kwargs)
            return result, {"error": "psutil not available"}
        
        process = psutil.Process()
        
        # 実行前のメモリ使用量
        memory_before = process.memory_info()
        
        # 関数実行
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # 実行後のメモリ使用量
        memory_after = process.memory_info()
        
        memory_metrics = {
            "rss_before_mb": memory_before.rss / 1024 / 1024,
            "rss_after_mb": memory_after.rss / 1024 / 1024,
            "rss_increase_mb": (memory_after.rss - memory_before.rss) / 1024 / 1024,
            "vms_before_mb": memory_before.vms / 1024 / 1024,
            "vms_after_mb": memory_after.vms / 1024 / 1024,
            "execution_time": end_time - start_time
        }
        
        return result, memory_metrics
    
    def calculate_throughput_metrics(
        self, 
        processing_times: List[float], 
        data_sizes: List[int]
    ) -> Dict[str, float]:
        """スループット指標の計算"""
        
        if len(processing_times) != len(data_sizes):
            raise ValueError("Processing times and data sizes must have same length")
        
        total_time = sum(processing_times)
        total_data = sum(data_sizes)
        
        metrics = {
            "total_processing_time": total_time,
            "total_data_processed": total_data,
            "average_processing_time": np.mean(processing_times),
            "processing_time_std": np.std(processing_times),
            "throughput_items_per_second": len(processing_times) / total_time if total_time > 0 else 0,
            "throughput_data_per_second": total_data / total_time if total_time > 0 else 0
        }
        
        # 効率性指標
        if data_sizes:
            efficiency_scores = [size / time if time > 0 else 0 
                               for size, time in zip(data_sizes, processing_times)]
            metrics.update({
                "efficiency_mean": np.mean(efficiency_scores),
                "efficiency_std": np.std(efficiency_scores),
                "efficiency_min": np.min(efficiency_scores),
                "efficiency_max": np.max(efficiency_scores)
            })
        
        return metrics
    
    def calculate_statistical_summary(self, values: List[float]) -> Dict[str, float]:
        """統計サマリーの計算"""
        if not values:
            return {"error": "Empty values list"}
        
        values_array = np.array(values)
        
        summary = {
            "count": len(values),
            "mean": np.mean(values_array),
            "median": np.median(values_array),
            "std": np.std(values_array),
            "var": np.var(values_array),
            "min": np.min(values_array),
            "max": np.max(values_array),
            "range": np.max(values_array) - np.min(values_array)
        }
        
        # パーセンタイル
        percentiles = [25, 75, 90, 95, 99]
        for p in percentiles:
            summary[f"percentile_{p}"] = np.percentile(values_array, p)
        
        # 外れ値検出（IQR法）
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = values_array[(values_array < lower_bound) | (values_array > upper_bound)]
        summary.update({
            "iqr": iqr,
            "outlier_count": len(outliers),
            "outlier_ratio": len(outliers) / len(values) if values else 0
        })
        
        return summary
    
    def benchmark_tokenization_speed(
        self, 
        tokenizer_func: Callable, 
        texts: List[str], 
        iterations: int = 3
    ) -> Dict[str, Any]:
        """トークン化速度ベンチマーク"""
        logger.info(f"Running tokenization benchmark with {len(texts)} texts, {iterations} iterations")
        
        results = {
            "text_count": len(texts),
            "iterations": iterations,
            "per_iteration_results": [],
            "overall_metrics": {}
        }
        
        all_times = []
        all_token_counts = []
        
        for iteration in range(iterations):
            iteration_results = {
                "iteration": iteration + 1,
                "text_results": [],
                "iteration_total_time": 0,
                "iteration_total_tokens": 0
            }
            
            for i, text in enumerate(texts):
                start_time = time.perf_counter()
                
                try:
                    tokens = tokenizer_func(text)
                    end_time = time.perf_counter()
                    
                    processing_time = end_time - start_time
                    token_count = len(tokens) if hasattr(tokens, '__len__') else 0
                    
                    text_result = {
                        "text_index": i,
                        "text_length": len(text),
                        "token_count": token_count,
                        "processing_time": processing_time,
                        "tokens_per_second": token_count / processing_time if processing_time > 0 else 0
                    }
                    
                    iteration_results["text_results"].append(text_result)
                    iteration_results["iteration_total_time"] += processing_time
                    iteration_results["iteration_total_tokens"] += token_count
                    
                    all_times.append(processing_time)
                    all_token_counts.append(token_count)
                    
                except Exception as e:
                    logger.error(f"Tokenization failed for text {i}: {e}")
                    text_result = {
                        "text_index": i,
                        "error": str(e)
                    }
                    iteration_results["text_results"].append(text_result)
            
            results["per_iteration_results"].append(iteration_results)
        
        # 全体統計の計算
        if all_times:
            results["overall_metrics"] = {
                "total_processing_time": sum(all_times),
                "average_time_per_text": np.mean(all_times),
                "time_std": np.std(all_times),
                "total_tokens": sum(all_token_counts),
                "average_tokens_per_text": np.mean(all_token_counts) if all_token_counts else 0,
                "overall_tokens_per_second": sum(all_token_counts) / sum(all_times) if sum(all_times) > 0 else 0
            }
            
            # スループット指標
            throughput_metrics = self.calculate_throughput_metrics(all_times, all_token_counts)
            results["overall_metrics"].update(throughput_metrics)
        
        return results
    
    def compare_performance_profiles(
        self, 
        profile1: Dict[str, float], 
        profile2: Dict[str, float],
        profile1_name: str = "Profile 1",
        profile2_name: str = "Profile 2"
    ) -> Dict[str, Any]:
        """性能プロファイルの比較"""
        
        comparison = {
            "profile1_name": profile1_name,
            "profile2_name": profile2_name,
            "comparisons": {},
            "summary": {}
        }
        
        # 共通メトリクスの比較
        common_metrics = set(profile1.keys()) & set(profile2.keys())
        
        improvements = []
        
        for metric in common_metrics:
            value1 = profile1[metric]
            value2 = profile2[metric]
            
            if value1 != 0:
                improvement_ratio = (value2 - value1) / value1
                improvement_percentage = improvement_ratio * 100
            else:
                improvement_ratio = float('inf') if value2 > 0 else 0
                improvement_percentage = float('inf') if value2 > 0 else 0
            
            comparison["comparisons"][metric] = {
                "value1": value1,
                "value2": value2,
                "difference": value2 - value1,
                "improvement_ratio": improvement_ratio,
                "improvement_percentage": improvement_percentage
            }
            
            if improvement_percentage != float('inf'):
                improvements.append(improvement_percentage)
        
        # サマリー統計
        if improvements:
            comparison["summary"] = {
                "average_improvement": np.mean(improvements),
                "median_improvement": np.median(improvements),
                "improvement_std": np.std(improvements),
                "positive_improvements": sum(1 for imp in improvements if imp > 0),
                "total_comparisons": len(improvements)
            }
        
        return comparison
    
    def generate_performance_report(
        self, 
        metrics: Dict[str, Any], 
        title: str = "Performance Analysis Report"
    ) -> str:
        """性能分析レポートの生成"""
        
        report_lines = [
            f"# {title}",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]
        
        # 基本メトリクス
        if "overall_metrics" in metrics:
            report_lines.extend([
                "## Overall Performance Metrics",
                ""
            ])
            
            for key, value in metrics["overall_metrics"].items():
                if isinstance(value, float):
                    report_lines.append(f"- {key}: {value:.3f}")
                else:
                    report_lines.append(f"- {key}: {value}")
            
            report_lines.append("")
        
        # 統計サマリー
        if "statistical_summary" in metrics:
            stats = metrics["statistical_summary"]
            report_lines.extend([
                "## Statistical Summary",
                "",
                f"- Count: {stats.get('count', 'N/A')}",
                f"- Mean: {stats.get('mean', 0):.3f}",
                f"- Median: {stats.get('median', 0):.3f}",
                f"- Standard Deviation: {stats.get('std', 0):.3f}",
                f"- Range: {stats.get('range', 0):.3f}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def save_metrics(self, metrics: Dict[str, Any], filename: Optional[str] = None):
        """メトリクスの保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        output_path = Path("evaluation_results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")

# ユーティリティ関数
def create_sample_tokenization_benchmark():
    """サンプルトークン化ベンチマーク"""
    
    def sample_tokenizer(text: str) -> List[str]:
        """サンプルトークナイザー（文字分割）"""
        return list(text)
    
    def advanced_tokenizer(text: str) -> List[str]:
        """改良トークナイザー（単語分割）"""
        return text.split()
    
    test_texts = [
        "機械学習による日本語処理の最適化",
        "DeepSeek R1モデルの性能評価と分析",
        "ROCm環境での効率的なGPU利用",
        "自然言語処理における深層学習の応用"
    ]
    
    metrics = PerformanceMetrics()
    
    # ベンチマーク実行
    result1 = metrics.benchmark_tokenization_speed(sample_tokenizer, test_texts)
    result2 = metrics.benchmark_tokenization_speed(advanced_tokenizer, test_texts)
    
    # 比較分析
    comparison = metrics.compare_performance_profiles(
        result1["overall_metrics"],
        result2["overall_metrics"],
        "Character Tokenizer",
        "Word Tokenizer"
    )
    
    return {
        "character_tokenizer": result1,
        "word_tokenizer": result2,
        "comparison": comparison
    }

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Metrics Calculator")
    parser.add_argument("--demo", action="store_true", 
                      help="Run demonstration benchmark")
    parser.add_argument("--output", help="Output filename")
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running demonstration benchmark...")
        results = create_sample_tokenization_benchmark()
        
        metrics = PerformanceMetrics()
        metrics.save_metrics(results, args.output)
        
        # レポート生成
        report = metrics.generate_performance_report(results, "Tokenization Benchmark Demo")
        print(report)
    else:
        print("Use --demo to run demonstration benchmark")

if __name__ == "__main__":
    main()
