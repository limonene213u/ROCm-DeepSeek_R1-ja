#!/usr/bin/env python3
"""
Comparative Analysis Script
Cross-model performance comparison and statistical analysis

Author: Akira Ito a.k.a limonene213u
複数モデルの性能比較と統計的有意性検定
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

# パス設定の修正
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import scipy.stats as stats
    from scipy.stats import ttest_rel, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Statistical tests will be limited.")
    SCIPY_AVAILABLE = False

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparativeAnalyzer:
    """モデル間比較分析クラス"""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.comparison_results = {}
        
    def load_evaluation_results(self, pattern: str = "*evaluation*.json") -> Dict[str, Any]:
        """評価結果ファイルの読み込み"""
        results = {}
        
        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[result_file.stem] = data
                    
                logger.info(f"Loaded results from {result_file}")
                
            except Exception as e:
                logger.error(f"Failed to load {result_file}: {e}")
        
        return results
    
    def compare_models(
        self, 
        results: Dict[str, Any], 
        baseline_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """モデル間性能比較"""
        logger.info("Starting model comparison analysis")
        
        # データの準備
        model_scores = self._extract_model_scores(results)
        
        if not model_scores:
            logger.warning("No valid model scores found")
            return {}
        
        # 基準モデルの設定
        if baseline_model is None:
            baseline_model = list(model_scores.keys())[0]
            logger.info(f"Using {baseline_model} as baseline")
        
        comparison_results = {
            "baseline_model": baseline_model,
            "comparisons": {},
            "summary_statistics": {},
            "rankings": {}
        }
        
        # 各モデルとの比較
        for model_name, scores in model_scores.items():
            if model_name == baseline_model:
                continue
                
            comparison = self._compare_model_pair(
                baseline_scores=model_scores[baseline_model],
                target_scores=scores,
                baseline_name=baseline_model,
                target_name=model_name
            )
            
            comparison_results["comparisons"][model_name] = comparison
        
        # 全体統計の計算
        comparison_results["summary_statistics"] = self._calculate_summary_statistics(model_scores)
        
        # ランキングの生成
        comparison_results["rankings"] = self._generate_rankings(model_scores)
        
        return comparison_results
    
    def _extract_model_scores(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """評価結果からスコアデータを抽出"""
        model_scores = {}
        
        for result_set_name, result_data in results.items():
            for model_name, model_data in result_data.items():
                if isinstance(model_data, dict) and "evaluation_scores" in model_data:
                    scores = model_data["evaluation_scores"]
                    if isinstance(scores, dict):
                        model_scores[model_name] = scores
        
        return model_scores
    
    def _compare_model_pair(
        self, 
        baseline_scores: Dict[str, float],
        target_scores: Dict[str, float],
        baseline_name: str,
        target_name: str
    ) -> Dict[str, Any]:
        """2モデル間の詳細比較"""
        
        # 共通タスクの特定
        common_tasks = set(baseline_scores.keys()) & set(target_scores.keys())
        
        if not common_tasks:
            return {"error": "No common tasks found"}
        
        # スコア配列の準備
        baseline_values = [baseline_scores[task] for task in common_tasks]
        target_values = [target_scores[task] for task in common_tasks]
        
        # 基本統計
        improvement_scores = [target - baseline for target, baseline in zip(target_values, baseline_values)]
        
        comparison = {
            "common_tasks": list(common_tasks),
            "baseline_mean": np.mean(baseline_values),
            "target_mean": np.mean(target_values),
            "improvement_mean": np.mean(improvement_scores),
            "improvement_std": np.std(improvement_scores),
            "relative_improvement": (np.mean(target_values) - np.mean(baseline_values)) / np.mean(baseline_values),
            "task_improvements": dict(zip(common_tasks, improvement_scores))
        }
        
        # 統計的有意性検定
        if SCIPY_AVAILABLE and len(baseline_values) > 1:
            try:
                # 対応ありt検定
                t_stat, p_value = ttest_rel(target_values, baseline_values)
                comparison["statistical_test"] = {
                    "test_type": "paired_t_test",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
                
                # 効果量の計算（Cohen's d）
                pooled_std = np.sqrt((np.var(baseline_values) + np.var(target_values)) / 2)
                if pooled_std > 0:
                    cohens_d = (np.mean(target_values) - np.mean(baseline_values)) / pooled_std
                    comparison["effect_size"] = {
                        "cohens_d": float(cohens_d),
                        "interpretation": self._interpret_effect_size(cohens_d)
                    }
                
            except Exception as e:
                logger.warning(f"Statistical test failed: {e}")
                comparison["statistical_test"] = {"error": str(e)}
        
        return comparison
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Cohen's d効果量の解釈"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_summary_statistics(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """全体統計の計算"""
        all_tasks = set()
        for scores in model_scores.values():
            all_tasks.update(scores.keys())
        
        summary = {
            "total_models": len(model_scores),
            "total_tasks": len(all_tasks),
            "task_coverage": {}
        }
        
        # タスク別統計
        for task in all_tasks:
            task_scores = []
            model_count = 0
            
            for model_name, scores in model_scores.items():
                if task in scores:
                    task_scores.append(scores[task])
                    model_count += 1
            
            if task_scores:
                summary["task_coverage"][task] = {
                    "models_evaluated": model_count,
                    "mean_score": np.mean(task_scores),
                    "std_score": np.std(task_scores),
                    "min_score": np.min(task_scores),
                    "max_score": np.max(task_scores)
                }
        
        return summary
    
    def _generate_rankings(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """モデルランキングの生成"""
        
        # 全体平均スコアによるランキング
        model_averages = {}
        for model_name, scores in model_scores.items():
            if scores:
                model_averages[model_name] = np.mean(list(scores.values()))
        
        # ランキングの作成
        ranked_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {
            "overall_ranking": [
                {"rank": i+1, "model": model, "average_score": score}
                for i, (model, score) in enumerate(ranked_models)
            ]
        }
        
        # タスク別ランキング
        all_tasks = set()
        for scores in model_scores.values():
            all_tasks.update(scores.keys())
        
        rankings["task_rankings"] = {}
        for task in all_tasks:
            task_scores = []
            for model_name, scores in model_scores.items():
                if task in scores:
                    task_scores.append((model_name, scores[task]))
            
            task_scores.sort(key=lambda x: x[1], reverse=True)
            rankings["task_rankings"][task] = [
                {"rank": i+1, "model": model, "score": score}
                for i, (model, score) in enumerate(task_scores)
            ]
        
        return rankings
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """比較レポートの生成"""
        report_lines = [
            "# Model Comparison Analysis Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"## Baseline Model: {comparison_results.get('baseline_model', 'Unknown')}",
            ""
        ]
        
        # 全体ランキング
        if "rankings" in comparison_results and "overall_ranking" in comparison_results["rankings"]:
            report_lines.extend([
                "## Overall Model Rankings",
                ""
            ])
            
            for item in comparison_results["rankings"]["overall_ranking"]:
                report_lines.append(
                    f"{item['rank']}. {item['model']}: {item['average_score']:.3f}"
                )
            
            report_lines.append("")
        
        # 個別比較結果
        if "comparisons" in comparison_results:
            report_lines.extend([
                "## Detailed Comparisons",
                ""
            ])
            
            for model_name, comparison in comparison_results["comparisons"].items():
                if "error" in comparison:
                    continue
                    
                report_lines.extend([
                    f"### {model_name}",
                    f"- Relative Improvement: {comparison['relative_improvement']:.1%}",
                    f"- Mean Improvement: {comparison['improvement_mean']:.3f}",
                    ""
                ])
                
                if "statistical_test" in comparison and "p_value" in comparison["statistical_test"]:
                    significance = "Significant" if comparison["statistical_test"]["significant"] else "Not significant"
                    report_lines.extend([
                        f"- Statistical Significance: {significance} (p={comparison['statistical_test']['p_value']:.3f})",
                        ""
                    ])
        
        return "\n".join(report_lines)
    
    def save_comparison_results(self, comparison_results: Dict[str, Any], filename: Optional[str] = None):
        """比較結果の保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
        
        output_path = self.results_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Comparison results saved to {output_path}")
        
        # レポートの生成と保存
        report = self.generate_comparison_report(comparison_results)
        report_path = output_path.with_suffix('.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Comparison report saved to {report_path}")

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Comparative Analysis")
    parser.add_argument("--results-dir", default="evaluation_results",
                      help="Directory containing evaluation results")
    parser.add_argument("--baseline", help="Baseline model name")
    parser.add_argument("--pattern", default="*evaluation*.json",
                      help="File pattern for evaluation results")
    
    args = parser.parse_args()
    
    analyzer = ComparativeAnalyzer(args.results_dir)
    
    # 評価結果の読み込み
    results = analyzer.load_evaluation_results(args.pattern)
    
    if not results:
        print("No evaluation results found")
        return
    
    # 比較分析の実行
    comparison_results = analyzer.compare_models(results, args.baseline)
    
    if comparison_results:
        # 結果の保存
        analyzer.save_comparison_results(comparison_results)
        print("Comparative analysis completed")
    else:
        print("No valid comparison results generated")

if __name__ == "__main__":
    main()
