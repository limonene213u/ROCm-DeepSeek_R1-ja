#!/usr/bin/env python3
"""
Scientific Framework Launch Script
科学的フレームワーク統合実行スクリプト

Author: Akira Ito a.k.a limonene213u
統合実行: 即座実装可能な最適化から完全な科学的パイプラインまで
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
import json

# 内部モジュールのインポート確認
try:
    from scientific_optimization_framework import JapaneseSpecializedModel, MI300XConfig, OptimizationLevel
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization framework not available: {e}")
    OPTIMIZATION_AVAILABLE = False

try:
    from vaporetto_integration import DeepSeekVaporettoIntegration
    VAPORETTO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vaporetto integration not available: {e}")
    VAPORETTO_AVAILABLE = False

try:
    from jlce_evaluation_system import JLCEEvaluator
    JLCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: JLCE evaluation not available: {e}")
    JLCE_AVAILABLE = False

try:
    from scientific_japanese_adaptation_pipeline import ScientificJapaneseAdaptationPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

MODULES_AVAILABLE = any([OPTIMIZATION_AVAILABLE, VAPORETTO_AVAILABLE, JLCE_AVAILABLE, PIPELINE_AVAILABLE])

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameworkLauncher:
    """科学的フレームワーク統合ランチャー"""
    
    def __init__(self):
        self.available_models = [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        ]
        
        self.execution_modes = {
            "quick": "即座実装最適化（5-10分）",
            "analysis": "分析・評価システム（15-30分）", 
            "full": "完全科学的パイプライン（60-120分）",
            "benchmark": "ベンチマーク・比較実行（30-60分）"
        }
    
    def show_menu(self):
        """メニュー表示"""
        print("\n" + "="*70)
        print("DEEPSEEK R1 SCIENTIFIC OPTIMIZATION FRAMEWORK")
        print("="*70)
        print("科学的最適化フレームワーク - 統合実行システム")
        print()
        
        print("実行モード:")
        for mode, description in self.execution_modes.items():
            print(f"  {mode}: {description}")
        print()
        
        print("対応モデル:")
        for i, model in enumerate(self.available_models, 1):
            model_name = model.split('/')[-1]
            print(f"  {i}. {model_name}")
        print()
    
    def get_user_selection(self) -> tuple:
        """ユーザー選択取得"""
        # 実行モード選択
        while True:
            mode = input("実行モード選択 (quick/analysis/full/benchmark): ").strip().lower()
            if mode in self.execution_modes:
                break
            print("無効な選択です。quick, analysis, full, benchmark から選択してください。")
        
        # モデル選択
        while True:
            try:
                print("\nモデル選択:")
                for i, model in enumerate(self.available_models, 1):
                    print(f"  {i}. {model.split('/')[-1]}")
                
                choice = input("モデル番号を選択 (1-4): ").strip()
                model_index = int(choice) - 1
                
                if 0 <= model_index < len(self.available_models):
                    selected_model = self.available_models[model_index]
                    break
                else:
                    print("無効な番号です。1-4から選択してください。")
            except ValueError:
                print("数字を入力してください。")
        
        return mode, selected_model
    
    async def run_quick_optimization(self, model_name: str):
        """即座実装最適化実行"""
        print(f"\n🚀 即座実装最適化を開始: {model_name}")
        
        if not OPTIMIZATION_AVAILABLE:
            print("❌ 最適化フレームワークが利用できません")
            print("📝 基本システム情報のみ表示:")
            print(f"  選択モデル: {model_name}")
            print(f"  Python版: {sys.version}")
            print(f"  利用可能モジュール: optimization={OPTIMIZATION_AVAILABLE}, vaporetto={VAPORETTO_AVAILABLE}")
            return
        
        try:
            # MI300X最適化設定
            config = MI300XConfig(optimization_level=OptimizationLevel.ADVANCED)
            
            # 日本語特化モデル初期化
            japanese_model = JapaneseSpecializedModel(model_name, config)
            
            # システム状態表示
            status = japanese_model.get_system_status()
            print("\n📊 システム状態:")
            print(f"  モデル: {status['model_name']}")
            print(f"  最適化レベル: {status['mi300x_config']['optimization_level']}")
            print(f"  ROCm環境: {status['rocm_environment'].get('torch_rocm_support', 'Unknown')}")
            
            # Vaporetto統合テスト
            if VAPORETTO_AVAILABLE:
                print("\n⚡ Vaporetto統合テスト実行中...")
                vaporetto = DeepSeekVaporettoIntegration(model_name)
                
                test_texts = [
                    "機械学習による日本語処理の最適化",
                    "DeepSeek R1モデルの性能評価"
                ]
                
                comparison = vaporetto.compare_tokenization_efficiency(test_texts)
                
                print("📈 最適化結果:")
                print(f"  処理速度向上: {comparison.get('speed_improvement', 'N/A'):.2f}x")
                print(f"  効率比率: {comparison.get('efficiency_ratio', 'N/A'):.2f}")
            else:
                print("⚠️  Vaporetto統合スキップ（モジュール未利用可能）")
            
            print("\n✅ 即座実装最適化完了!")
            
        except Exception as e:
            print(f"❌ 最適化エラー: {e}")
            logger.error(f"Quick optimization failed: {e}")
    
    async def run_analysis_system(self, model_name: str):
        """分析・評価システム実行"""
        print(f"\n🔬 分析・評価システムを開始: {model_name}")
        
        if not JLCE_AVAILABLE:
            print("❌ JLCE評価システムが利用できません")
            print("📝 基本分析のみ実行:")
            
            # 基本システム分析
            if OPTIMIZATION_AVAILABLE:
                try:
                    config = MI300XConfig()
                    japanese_model = JapaneseSpecializedModel(model_name, config)
                    status = japanese_model.get_system_status()
                    
                    print("\n📊 システム状態:")
                    print(f"  モデル: {status['model_name']}")
                    print(f"  メモリ使用量: {status.get('memory_usage', 'Unknown')}")
                    print(f"  最適化状態: {status.get('optimization_status', 'Unknown')}")
                except Exception as e:
                    print(f"⚠️  基本分析エラー: {e}")
            
            if VAPORETTO_AVAILABLE:
                try:
                    print("\n⚡ トークン化効率テスト:")
                    vaporetto = DeepSeekVaporettoIntegration(model_name)
                    test_texts = ["科学的最適化の評価と分析", "日本語言語モデルの性能測定"]
                    comparison = vaporetto.compare_tokenization_efficiency(test_texts)
                    print(f"  速度向上: {comparison.get('speed_improvement', 'N/A'):.2f}x")
                except Exception as e:
                    print(f"⚠️  トークン化テストエラー: {e}")
            
            print("\n✅ 基本分析完了!")
            return
        
        try:
            # JLCE評価システム
            evaluator = JLCEEvaluator()
            
            # サンプルデータでの評価テスト
            from jlce_evaluation_system import create_sample_test_data
            test_datasets = create_sample_test_data()
            
            print(f"📝 評価タスク数: {len(test_datasets)}")
            print("📊 評価タスク:")
            for task_name in test_datasets.keys():
                print(f"  - {task_name}")
            
            # 統合分析実行
            if VAPORETTO_AVAILABLE:
                vaporetto = DeepSeekVaporettoIntegration(model_name)
                
                test_texts = [
                    "日本語の自然言語処理における課題",
                    "機械学習モデルの評価と最適化手法",
                    "文化的コンテキストを考慮した言語理解"
                ]
                
                # 日本語特性分析
                characteristics = vaporetto.analyze_japanese_characteristics(test_texts)
                
                print("\n📈 日本語特性分析結果:")
                char_dist = characteristics["character_distribution"]
                for script, stats in char_dist.items():
                    print(f"  {script}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            print("\n✅ 分析・評価システム完了!")
            
        except Exception as e:
            print(f"❌ 分析エラー: {e}")
            logger.error(f"Analysis system failed: {e}")
    
    async def run_full_pipeline(self, model_name: str):
        """完全科学的パイプライン実行"""
        print(f"\n🧪 完全科学的パイプラインを開始: {model_name}")
        
        if not MODULES_AVAILABLE:
            print("Error: Required modules not available")
            return
        
        try:
            # 科学的日本語特化パイプライン初期化
            pipeline = ScientificJapaneseAdaptationPipeline(
                model_name=model_name,
                output_dir="scientific_pipeline_results",
                enable_experimental=False
            )
            
            print("🔄 最適化サイクル実行中...")
            print("  Stage 1: 初期解析 (5分)")
            print("  Stage 2: 深層解析 (15分)")
            print("  Stage 3: 戦略策定 (10分)")
            print("  Stage 4: 実装・評価 (継続)")
            
            # パイプライン実行
            report = await pipeline.execute_optimization_cycle()
            
            print("\n📊 パイプライン結果:")
            print(f"  Pipeline ID: {report.pipeline_id}")
            print(f"  総実行時間: {report.total_duration:.2f}秒")
            print(f"  成功率: {report.overall_success_rate:.2%}")
            
            print("\n📈 期待される改善:")
            if report.adaptation_strategy:
                for metric, value in report.adaptation_strategy.expected_improvements.items():
                    print(f"  {metric}: {value:.1f}x")
            
            print(f"\n📁 詳細レポート: {pipeline.output_dir}")
            print("\n✅ 完全科学的パイプライン完了!")
            
        except Exception as e:
            print(f"❌ パイプラインエラー: {e}")
    
    async def run_benchmark(self, model_name: str):
        """ベンチマーク・比較実行"""
        print(f"\n⏱️ ベンチマーク・比較を開始: {model_name}")
        
        try:
            # 複数システムのベンチマーク
            print("🔄 ベンチマーク実行中...")
            
            results = {
                "model": model_name,
                "timestamp": pd.Timestamp.now().isoformat(),
                "benchmarks": {}
            }
            
            if MODULES_AVAILABLE:
                # Vaporetto性能ベンチマーク
                vaporetto = DeepSeekVaporettoIntegration(model_name)
                
                test_texts = [
                    "機械学習による自然言語処理の研究",
                    "深層学習モデルの日本語対応最適化",
                    "ニューラルネットワークの性能評価",
                    "トランスフォーマーアーキテクチャの解析",
                    "言語モデルの文化的適応性向上"
                ]
                
                comparison = vaporetto.compare_tokenization_efficiency(test_texts)
                results["benchmarks"]["tokenization"] = comparison
                
                print("📊 トークナイゼーション性能:")
                print(f"  Vaporetto速度: {comparison['vaporetto']['tokens_per_second']:.2f} tokens/sec")
                print(f"  DeepSeek速度: {comparison['deepseek']['tokens_per_second']:.2f} tokens/sec")
                print(f"  改善率: {comparison.get('efficiency_ratio', 1):.2f}x")
            
            # システム情報
            if MODULES_AVAILABLE:
                config = MI300XConfig()
                japanese_model = JapaneseSpecializedModel(model_name, config)
                system_status = japanese_model.get_system_status()
                results["system_status"] = system_status
            
            # 結果保存
            output_file = f"benchmark_results_{int(time.time())}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n📁 ベンチマーク結果保存: {output_file}")
            print("\n✅ ベンチマーク・比較完了!")
            
        except Exception as e:
            print(f"❌ ベンチマークエラー: {e}")

async def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="DeepSeek R1 Scientific Optimization Framework")
    parser.add_argument("--mode", choices=["quick", "analysis", "full", "benchmark"], 
                       help="実行モード")
    parser.add_argument("--model", type=int, choices=[1, 2, 3, 4], 
                       help="モデル選択 (1-4)")
    parser.add_argument("--interactive", action="store_true", 
                       help="対話形式で実行")
    
    args = parser.parse_args()
    
    launcher = FrameworkLauncher()
    
    if args.interactive or (not args.mode or not args.model):
        # 対話モード
        launcher.show_menu()
        mode, model_name = launcher.get_user_selection()
    else:
        # コマンドライン引数モード
        mode = args.mode
        model_name = launcher.available_models[args.model - 1]
    
    print(f"\n🎯 実行設定:")
    print(f"  モード: {mode} - {launcher.execution_modes[mode]}")
    print(f"  モデル: {model_name.split('/')[-1]}")
    
    # 確認
    if not args.interactive:
        confirm = input("\n実行しますか? (y/N): ").strip().lower()
        if confirm != 'y':
            print("実行をキャンセルしました。")
            return
    
    # 実行
    print(f"\n🚀 {launcher.execution_modes[mode]} を開始...")
    
    if mode == "quick":
        await launcher.run_quick_optimization(model_name)
    elif mode == "analysis":
        await launcher.run_analysis_system(model_name)
    elif mode == "full":
        await launcher.run_full_pipeline(model_name)
    elif mode == "benchmark":
        await launcher.run_benchmark(model_name)
    
    print("\n🎉 フレームワーク実行完了!")
    print("詳細結果は生成されたファイルを参照してください。")

if __name__ == "__main__":
    import time
    import pandas as pd
    asyncio.run(main())
