#!/usr/bin/env python3
"""
DeepSeek R1 モデル解析ツール（軽量版）
BPE（SentencePiece）トークナイザーの日本語対応状況の詳細分析
※可視化機能を除いた基本版

Author: Akira Ito a.k.a limonene213u
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
import re
import unicodedata

# Transformers関連
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TokenAnalysisResult:
    """トークン解析結果を格納するデータクラス"""
    model_name: str
    vocab_size: int
    japanese_tokens: List[str]
    japanese_token_count: int
    japanese_ratio: float
    hiragana_tokens: List[str]
    katakana_tokens: List[str]
    kanji_tokens: List[str]
    mixed_tokens: List[str]
    byte_tokens: List[str]
    subword_efficiency: Dict[str, float]
    common_word_coverage: Dict[str, bool]
    token_length_stats: Dict[str, float]

class DeepSeekR1AnalyzerLite:
    """DeepSeek R1 モデルのBPE解析クラス（軽量版）"""
    
    def __init__(self, output_dir: str = "Analyze_DeepSeekR1_Data"):
        """解析器の初期化"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 解析対象モデルの定義
        self.target_models = [
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        ]
        
        # 日本語テストサンプル
        self.japanese_test_sentences = [
            "こんにちは、世界！",
            "日本語の自然言語処理は難しいです。",
            "機械学習による言語モデルの訓練",
            "トークナイザーのサブワード分割",
            "ひらがな、カタカナ、漢字の混在",
            "研究開発における技術的課題",
            "人工知能の発展と社会への影響",
            "データサイエンスの実践的応用"
        ]
        
        # 日本語の一般的な単語リスト
        self.common_japanese_words = [
            "こんにちは", "ありがとう", "すみません", "はじめまして",
            "学習", "研究", "開発", "技術", "科学", "大学",
            "会社", "仕事", "問題", "解決", "方法", "結果",
            "データ", "情報", "システム", "プログラム"
        ]
        
        logger.info(f"DeepSeek R1 Analyzer (Lite) initialized. Output: {self.output_dir}")
    
    def analyze_tokenizer(self, model_name: str) -> TokenAnalysisResult:
        """指定されたモデルのトークナイザーを詳細分析"""
        logger.info(f"Analyzing tokenizer for: {model_name}")
        
        try:
            # トークナイザーの読み込み
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # 語彙の取得
            vocab = tokenizer.get_vocab()
            vocab_size = len(vocab)
            
            logger.info(f"Vocabulary size: {vocab_size}")
            
            # 日本語トークンの分析
            japanese_analysis = self._analyze_japanese_tokens(vocab)
            
            # サブワード効率の分析
            subword_efficiency = self._analyze_subword_efficiency(tokenizer)
            
            # 一般的な単語のカバレッジ分析
            common_word_coverage = self._analyze_common_word_coverage(tokenizer)
            
            # トークン長の統計
            token_length_stats = self._analyze_token_lengths(vocab)
            
            # 結果をまとめる
            result = TokenAnalysisResult(
                model_name=model_name,
                vocab_size=vocab_size,
                japanese_tokens=japanese_analysis['all_japanese'],
                japanese_token_count=len(japanese_analysis['all_japanese']),
                japanese_ratio=len(japanese_analysis['all_japanese']) / vocab_size,
                hiragana_tokens=japanese_analysis['hiragana'],
                katakana_tokens=japanese_analysis['katakana'],
                kanji_tokens=japanese_analysis['kanji'],
                mixed_tokens=japanese_analysis['mixed'],
                byte_tokens=japanese_analysis['byte_tokens'],
                subword_efficiency=subword_efficiency,
                common_word_coverage=common_word_coverage,
                token_length_stats=token_length_stats
            )
            
            logger.info(f"Analysis completed. Japanese tokens: {result.japanese_token_count} ({result.japanese_ratio:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {model_name}: {e}")
            raise
    
    def _analyze_japanese_tokens(self, vocab: Dict[str, int]) -> Dict[str, List[str]]:
        """語彙内の日本語トークンを分類分析"""
        hiragana_tokens = []
        katakana_tokens = []
        kanji_tokens = []
        mixed_tokens = []
        byte_tokens = []
        all_japanese = []
        
        for token in vocab.keys():
            try:
                # バイトトークンの判定
                if token.startswith('<0x') or (len(token) == 6 and token.startswith('<') and token.endswith('>')):
                    byte_tokens.append(token)
                    continue
                
                # 特殊トークンのスキップ
                if token.startswith('<') and token.endswith('>'):
                    continue
                
                # Unicode正規化
                normalized_token = unicodedata.normalize('NFKC', token)
                
                # 日本語文字の判定
                has_hiragana = bool(re.search(r'[\u3040-\u309F]', normalized_token))
                has_katakana = bool(re.search(r'[\u30A0-\u30FF]', normalized_token))
                has_kanji = bool(re.search(r'[\u4E00-\u9FAF]', normalized_token))
                
                if has_hiragana or has_katakana or has_kanji:
                    all_japanese.append(token)
                    
                    if has_hiragana and not has_katakana and not has_kanji:
                        hiragana_tokens.append(token)
                    elif has_katakana and not has_hiragana and not has_kanji:
                        katakana_tokens.append(token)
                    elif has_kanji and not has_hiragana and not has_katakana:
                        kanji_tokens.append(token)
                    else:
                        mixed_tokens.append(token)
                        
            except Exception:
                continue
        
        return {
            'all_japanese': all_japanese,
            'hiragana': hiragana_tokens,
            'katakana': katakana_tokens,
            'kanji': kanji_tokens,
            'mixed': mixed_tokens,
            'byte_tokens': byte_tokens
        }
    
    def _analyze_subword_efficiency(self, tokenizer) -> Dict[str, float]:
        """日本語テキストでのサブワード分割効率を分析"""
        total_chars = 0
        total_tokens = 0
        compression_ratios = []
        
        for sentence in self.japanese_test_sentences:
            char_count = len(sentence)
            tokens = tokenizer.tokenize(sentence)
            token_count = len(tokens)
            
            total_chars += char_count
            total_tokens += token_count
            
            if char_count > 0:
                compression_ratios.append(token_count / char_count)
        
        return {
            'avg_compression_ratio': float(total_tokens / total_chars if total_chars > 0 else 0),
            'compression_ratio_std': float(np.std(compression_ratios)),
            'avg_tokens_per_sentence': float(total_tokens / len(self.japanese_test_sentences)),
            'avg_chars_per_token': float(total_chars / total_tokens if total_tokens > 0 else 0)
        }
    
    def _analyze_common_word_coverage(self, tokenizer) -> Dict[str, bool]:
        """一般的な日本語単語のトークナイザーカバレッジを分析"""
        coverage = {}
        vocab = set(tokenizer.get_vocab().keys())
        
        for word in self.common_japanese_words:
            # 単語がそのまま語彙に含まれているか
            direct_match = word in vocab
            
            # トークン化して単一トークンになるか
            tokens = tokenizer.tokenize(word)
            single_token = len(tokens) == 1 and tokens[0] == word
            
            coverage[word] = direct_match or single_token
        
        return coverage
    
    def _analyze_token_lengths(self, vocab: Dict[str, int]) -> Dict[str, float]:
        """トークンの長さ統計を分析"""
        lengths = []
        for token in vocab.keys():
            try:
                # 特殊トークンをスキップ
                if not (token.startswith('<') and token.endswith('>')):
                    lengths.append(len(token))
            except Exception:
                continue
        
        if not lengths:
            return {}
        
        return {
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
            'min_length': float(np.min(lengths)),
            'max_length': float(np.max(lengths)),
            'q25_length': float(np.percentile(lengths, 25)),
            'q75_length': float(np.percentile(lengths, 75))
        }
    
    def compare_models(self, models_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """全ての対象モデルを比較分析"""
        target_models = models_subset or self.target_models
        logger.info(f"Starting model comparison for {len(target_models)} models...")
        
        results = []
        for model_name in target_models:
            try:
                result = self.analyze_tokenizer(model_name)
                
                # データフレーム用のフラットな辞書に変換
                flat_result = {
                    'model_name': result.model_name.split('/')[-1],
                    'vocab_size': result.vocab_size,
                    'japanese_token_count': result.japanese_token_count,
                    'japanese_ratio': result.japanese_ratio,
                    'hiragana_count': len(result.hiragana_tokens),
                    'katakana_count': len(result.katakana_tokens),
                    'kanji_count': len(result.kanji_tokens),
                    'mixed_count': len(result.mixed_tokens),
                    'byte_token_count': len(result.byte_tokens),
                    'avg_compression_ratio': result.subword_efficiency.get('avg_compression_ratio', 0),
                    'compression_ratio_std': result.subword_efficiency.get('compression_ratio_std', 0),
                    'avg_tokens_per_sentence': result.subword_efficiency.get('avg_tokens_per_sentence', 0),
                    'avg_chars_per_token': result.subword_efficiency.get('avg_chars_per_token', 0),
                    'common_word_coverage_rate': sum(result.common_word_coverage.values()) / len(result.common_word_coverage),
                    'mean_token_length': result.token_length_stats.get('mean_length', 0),
                    'median_token_length': result.token_length_stats.get('median_token_length', 0)
                }
                
                results.append(flat_result)
                
                # 個別の詳細結果も保存
                self._save_detailed_result(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze {model_name}: {e}")
                continue
        
        # 比較データフレームの作成
        comparison_df = pd.DataFrame(results)
        
        # 結果の保存
        output_path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Comparison results saved to: {output_path}")
        
        return comparison_df
    
    def _save_detailed_result(self, result: TokenAnalysisResult):
        """詳細な解析結果をJSONファイルに保存"""
        model_name = result.model_name.split('/')[-1]
        output_path = self.output_dir / f"{model_name}_detailed_analysis.json"
        
        # JSONシリアライズ可能な形式に変換
        detailed_data = {
            'model_name': result.model_name,
            'vocab_size': result.vocab_size,
            'japanese_analysis': {
                'total_count': result.japanese_token_count,
                'ratio': result.japanese_ratio,
                'by_type': {
                    'hiragana': {
                        'count': len(result.hiragana_tokens),
                        'examples': result.hiragana_tokens[:20]
                    },
                    'katakana': {
                        'count': len(result.katakana_tokens),
                        'examples': result.katakana_tokens[:20]
                    },
                    'kanji': {
                        'count': len(result.kanji_tokens),
                        'examples': result.kanji_tokens[:20]
                    },
                    'mixed': {
                        'count': len(result.mixed_tokens),
                        'examples': result.mixed_tokens[:20]
                    }
                }
            },
            'subword_efficiency': result.subword_efficiency,
            'common_word_coverage': result.common_word_coverage,
            'token_length_stats': result.token_length_stats,
            'byte_tokens_count': len(result.byte_tokens)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Detailed analysis saved to: {output_path}")
    
    def generate_summary_report(self, comparison_df: pd.DataFrame):
        """解析結果のサマリーレポートを生成"""
        logger.info("Generating summary report...")
        
        report_path = self.output_dir / "analysis_summary_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# DeepSeek R1 Models: Japanese Tokenizer Analysis Report\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes the Japanese language support in DeepSeek R1 Distill models' tokenizers.\n\n")
            
            f.write("## Model Comparison Overview\n\n")
            f.write("```\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n```\n\n")
            
            if not comparison_df.empty:
                f.write("## Key Findings\n\n")
                
                # 最も日本語対応が良いモデル
                best_jp_model = comparison_df.loc[comparison_df['japanese_ratio'].idxmax(), 'model_name']
                best_jp_ratio = comparison_df['japanese_ratio'].max()
                f.write(f"- **Best Japanese Support**: {best_jp_model} ({best_jp_ratio:.3f} Japanese token ratio)\n")
                
                # 最も効率的な圧縮
                best_compression_model = comparison_df.loc[comparison_df['avg_compression_ratio'].idxmin(), 'model_name']
                best_compression_ratio = comparison_df['avg_compression_ratio'].min()
                f.write(f"- **Most Efficient Compression**: {best_compression_model} ({best_compression_ratio:.3f} tokens/char)\n")
                
                # 最良のカバレッジ
                best_coverage_model = comparison_df.loc[comparison_df['common_word_coverage_rate'].idxmax(), 'model_name']
                best_coverage_rate = comparison_df['common_word_coverage_rate'].max()
                f.write(f"- **Best Common Word Coverage**: {best_coverage_model} ({best_coverage_rate:.3f} coverage rate)\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the analysis results:\n\n")
            f.write("1. For Japanese text processing, consider the model with highest Japanese token ratio\n")
            f.write("2. For efficiency, consider the model with lowest compression ratio\n")
            f.write("3. For general Japanese usage, consider the model with highest common word coverage\n")
        
        logger.info(f"Summary report saved to: {report_path}")


def main():
    """メイン実行関数"""
    logger.info("Starting DeepSeek R1 Japanese tokenizer analysis (Lite version)...")
    
    # 解析器の初期化
    analyzer = DeepSeekR1AnalyzerLite()
    
    try:
        # テスト用に1つのモデルのみ解析
        test_models = ["deepseek-ai/deepseek-r1-distill-qwen-1.5b"]
        
        # モデル比較分析実行
        comparison_df = analyzer.compare_models(test_models)
        
        # サマリーレポート生成
        analyzer.generate_summary_report(comparison_df)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved in: {analyzer.output_dir}")
        
        # 結果表示
        if not comparison_df.empty:
            print("\n" + "="*50)
            print("ANALYSIS SUMMARY")
            print("="*50)
            print(comparison_df[['model_name', 'vocab_size', 'japanese_ratio', 'avg_compression_ratio', 'common_word_coverage_rate']].to_string(index=False))
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
