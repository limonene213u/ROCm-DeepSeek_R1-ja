#!/usr/bin/env python3
"""
R-2 Swallow推論効率測定 テストスクリプト

実装完了確認とローカル動作テスト用
"""

import sys
import json
from pathlib import Path

# プロジェクトパス追加
sys.path.append(str(Path(__file__).parent.parent))

try:
    from Python.Benchmark.swallow_inference_benchmark import SwallowInferenceBenchmark
    print("✅ SwallowInferenceBenchmark import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_prompt_loading():
    """プロンプトファイル読み込みテスト"""
    prompt_file = "dataset/prompts_swallow_bench.jsonl"
    
    if not Path(prompt_file).exists():
        print(f"❌ Prompt file not found: {prompt_file}")
        return False
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = [json.loads(line.strip()) for line in f]
        
        print(f"✅ Loaded {len(prompts)} prompts from {prompt_file}")
        print(f"   First prompt: {prompts[0]['prompt'][:50]}...")
        return True
    
    except Exception as e:
        print(f"❌ Prompt loading failed: {e}")
        return False

def test_benchmark_initialization():
    """ベンチマーククラス初期化テスト"""
    try:
        benchmark = SwallowInferenceBenchmark(results_dir="test_results")
        print("✅ SwallowInferenceBenchmark initialized successfully")
        
        # 必要メソッドの存在確認
        required_methods = [
            'load_prompts',
            'bootstrap_confidence_interval', 
            'run_comparative_benchmark',
            'print_summary'
        ]
        
        for method in required_methods:
            if hasattr(benchmark, method):
                print(f"   ✅ Method {method} found")
            else:
                print(f"   ❌ Method {method} missing")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ Benchmark initialization failed: {e}")
        return False

def test_validation_runner_integration():
    """検証ランナー統合テスト"""
    try:
        from Python.Validation.paper_validation_runner import PaperValidationRunner
        
        runner = PaperValidationRunner(output_dir="test_validation_results")
        print("✅ PaperValidationRunner initialized successfully")
        
        # R-2検証メソッドの存在確認
        if hasattr(runner, 'validate_r2_swallow_efficiency'):
            print("   ✅ validate_r2_swallow_efficiency method found")
            return True
        else:
            print("   ❌ validate_r2_swallow_efficiency method missing")
            return False
    
    except Exception as e:
        print(f"❌ ValidationRunner integration failed: {e}")
        return False

def test_mock_benchmark_run():
    """モック環境でのベンチマーク実行テスト"""
    try:
        benchmark = SwallowInferenceBenchmark(results_dir="test_results")
        
        # プロンプト読み込み
        prompt_file = "dataset/prompts_swallow_bench.jsonl"
        if not Path(prompt_file).exists():
            print("❌ Skipping mock benchmark test - prompts not found")
            return False
        
        prompts = benchmark.load_prompts(prompt_file)
        print(f"✅ Mock test: Loaded {len(prompts)} prompts")
        
        # 信頼区間計算テスト
        test_data = [1.0, 1.2, 0.9, 1.1, 1.05]
        ci = benchmark.bootstrap_confidence_interval(test_data)
        print(f"✅ Mock test: Confidence interval calculation works: {ci}")
        
        return True
    
    except Exception as e:
        print(f"❌ Mock benchmark test failed: {e}")
        return False

def main():
    """テスト実行メイン"""
    print("="*50)
    print("R-2 Swallow Inference Efficiency Implementation Test")
    print("="*50)
    
    tests = [
        ("Prompt Loading", test_prompt_loading),
        ("Benchmark Initialization", test_benchmark_initialization),
        ("ValidationRunner Integration", test_validation_runner_integration),
        ("Mock Benchmark Run", test_mock_benchmark_run)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed: {test_name}")
    
    print(f"\n" + "="*50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! R-2 implementation ready.")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    
    print("="*50)

if __name__ == "__main__":
    main()
