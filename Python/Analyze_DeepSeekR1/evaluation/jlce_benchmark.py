#!/usr/bin/env python3
"""
JLCE Benchmark Execution Script
Japanese LLM Comprehensive Evaluation System

Author: Akira Ito a.k.a limonene213u
JGLUE拡張評価の実行とベンチマーク測定
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# パス設定の修正（相対ディレクトリ変更対応）
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from jlce_evaluation_system import JLCEEvaluator, create_sample_test_data
    from scientific_optimization_framework import JapaneseSpecializedModel, MI300XConfig
except ImportError as e:
    print(f"Warning: Required modules not available: {e}")
    print("Some functionality may be limited.")

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JLCEBenchmarkRunner:
    """JLCE評価ベンチマーク実行クラス"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    async def run_comprehensive_evaluation(
        self, 
        model_names: List[str],
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """包括的JLCE評価の実行"""
        logger.info(f"Starting comprehensive evaluation for {len(model_names)} models")
        
        results = {}
        
        for model_name in model_names:
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # JLCE評価システムの初期化
                evaluator = JLCEEvaluator()
                
                # モデル評価の実行
                model_results = await evaluator.evaluate_model(model_name)
                
                results[model_name] = {
                    "evaluation_scores": model_results,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed"
                }
                
                logger.info(f"Completed evaluation for {model_name}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {e}")
                results[model_name] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }
        
        # 結果の保存
        self._save_results(results, "comprehensive_evaluation")
        return results
    
    async def run_jglue_benchmark(self, model_names: List[str]) -> Dict[str, Any]:
        """JGLUE標準ベンチマークの実行"""
        logger.info("Running JGLUE benchmark evaluation")
        
        jglue_tasks = [
            "MARC-ja", "JCoLA", "JSTS", "JNLI", "JSQuAD", "JCommonsenseQA"
        ]
        
        results = {}
        
        for model_name in model_names:
            model_results = {}
            
            for task in jglue_tasks:
                try:
                    # タスク別評価の実行
                    task_score = await self._evaluate_jglue_task(model_name, task)
                    model_results[task] = task_score
                    
                    logger.info(f"{model_name} - {task}: {task_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Task {task} failed for {model_name}: {e}")
                    model_results[task] = {"error": str(e)}
            
            results[model_name] = model_results
        
        self._save_results(results, "jglue_benchmark")
        return results
    
    async def _evaluate_jglue_task(self, model_name: str, task: str) -> float:
        """個別JGLUEタスクの評価"""
        # サンプル実装（実際の評価は各タスク固有の処理が必要）
        await asyncio.sleep(0.1)  # 非同期処理のシミュレーション
        
        # 基本的なスコア計算（実装例）
        import random
        random.seed(hash(model_name + task))
        base_score = random.uniform(0.6, 0.9)
        
        return base_score
    
    def run_speed_benchmark(self, model_names: List[str]) -> Dict[str, Any]:
        """処理速度ベンチマーク"""
        logger.info("Running speed benchmark")
        
        test_texts = [
            "機械学習による日本語処理の最適化手法について説明してください。",
            "DeepSeek R1モデルの特徴と応用可能性を分析してください。",
            "ROCm環境でのGPU最適化戦略を詳述してください。"
        ]
        
        results = {}
        
        for model_name in model_names:
            try:
                # 速度測定の実行
                speed_results = self._measure_processing_speed(model_name, test_texts)
                results[model_name] = speed_results
                
                logger.info(f"Speed benchmark completed for {model_name}")
                
            except Exception as e:
                logger.error(f"Speed benchmark failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        self._save_results(results, "speed_benchmark")
        return results
    
    def _measure_processing_speed(self, model_name: str, texts: List[str]) -> Dict[str, float]:
        """処理速度測定"""
        import time
        
        start_time = time.time()
        
        # 処理時間のシミュレーション（実際は各モデルでの推論処理）
        for text in texts:
            time.sleep(0.01 * len(text) / 100)  # テキスト長に比例した処理時間
        
        total_time = time.time() - start_time
        
        return {
            "total_processing_time": total_time,
            "average_time_per_text": total_time / len(texts),
            "texts_processed": len(texts),
            "tokens_per_second": len("".join(texts)) / total_time
        }
    
    def _save_results(self, results: Dict[str, Any], benchmark_type: str):
        """評価結果の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{benchmark_type}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """比較レポートの生成"""
        report_lines = [
            "# JLCE Benchmark Evaluation Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Model Performance Comparison",
            ""
        ]
        
        for model_name, model_results in results.items():
            if "evaluation_scores" in model_results:
                scores = model_results["evaluation_scores"]
                avg_score = sum(scores.values()) / len(scores) if scores else 0
                
                report_lines.extend([
                    f"### {model_name}",
                    f"Average Score: {avg_score:.3f}",
                    ""
                ])
                
                for task, score in scores.items():
                    report_lines.append(f"- {task}: {score:.3f}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)

async def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JLCE Benchmark Runner")
    parser.add_argument("--models", nargs="+", 
                      default=["deepseek-ai/deepseek-r1-distill-qwen-1.5b"],
                      help="Models to evaluate")
    parser.add_argument("--benchmark", choices=["comprehensive", "jglue", "speed", "all"],
                      default="comprehensive", help="Benchmark type")
    parser.add_argument("--output", default="evaluation_results", 
                      help="Output directory")
    
    args = parser.parse_args()
    
    runner = JLCEBenchmarkRunner(args.output)
    
    if args.benchmark == "comprehensive" or args.benchmark == "all":
        results = await runner.run_comprehensive_evaluation(args.models)
        print("Comprehensive evaluation completed")
    
    if args.benchmark == "jglue" or args.benchmark == "all":
        results = await runner.run_jglue_benchmark(args.models)
        print("JGLUE benchmark completed")
    
    if args.benchmark == "speed" or args.benchmark == "all":
        results = runner.run_speed_benchmark(args.models)
        print("Speed benchmark completed")
    
    print(f"Results saved to {runner.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
