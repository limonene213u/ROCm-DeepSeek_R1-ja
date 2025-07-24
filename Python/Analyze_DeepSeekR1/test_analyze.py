#!/usr/bin/env python3
"""
DeepSeek R1 BPE解析ツール テスト実行スクリプト
軽量テスト用に一部機能を制限した実行モード

Author: Akira Ito a.k.a limonene213u
"""

import sys
import os
from pathlib import Path

# 相対パスでのインポート設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "Python" / "Analyze_DeepSeekR1"))

try:
    from analyze_deepseekr1 import DeepSeekR1Analyzer, logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install torch transformers datasets numpy pandas sentencepiece")
    sys.exit(1)

def test_single_model():
    """単一モデルのテスト実行"""
    print("Testing single model analysis...")
    
    analyzer = DeepSeekR1Analyzer("Analyze_DeepSeekR1_Data")
    
    # テスト用に最軽量モデルのみ解析
    test_model = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
    
    try:
        result = analyzer.analyze_tokenizer(test_model)
        
        print(f"\nTest Results for {test_model}:")
        print(f"Vocabulary Size: {result.vocab_size}")
        print(f"Japanese Tokens: {result.japanese_token_count}")
        print(f"Japanese Ratio: {result.japanese_ratio:.4f}")
        print(f"Hiragana Tokens: {len(result.hiragana_tokens)}")
        print(f"Katakana Tokens: {len(result.katakana_tokens)}")
        print(f"Kanji Tokens: {len(result.kanji_tokens)}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_tokenizer_only():
    """トークナイザーの基本機能のみテスト"""
    print("Testing basic tokenizer functionality...")
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        test_text = "こんにちは、世界！"
        tokens = tokenizer.tokenize(test_text)
        
        print(f"Test text: {test_text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
        
        print("\nBasic tokenizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Basic test failed: {e}")
        return False

def main():
    """メイン実行関数"""
    print("DeepSeek R1 BPE Analysis Tool - Test Mode")
    print("="*50)
    
    # 基本テスト
    if not test_tokenizer_only():
        print("Basic tokenizer test failed. Please check dependencies.")
        return False
    
    print("\n" + "-"*30)
    
    # 詳細テスト
    if not test_single_model():
        print("Single model analysis test failed.")
        return False
    
    print("\n" + "="*50)
    print("All tests passed! You can now run the full analysis.")
    print("To run full analysis: python Python/Analyze_DeepSeekR1/analyze_deepseekr1.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
