#!/usr/bin/env python3
"""
データセット品質向上・整形スクリプト
DeepSeek R1日本語特化学習用データセットの品質管理

Author: limonene213u
Purpose: 現行データセットの品質向上と適切な形式への整形
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging
from dataclasses import dataclass
import unicodedata

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityConfig:
    """データ品質設定"""
    min_text_length: int = 10
    max_text_length: int = 2000
    min_japanese_ratio: float = 0.3
    filter_adult_content: bool = True
    filter_spam: bool = True
    normalize_unicode: bool = True
    remove_duplicates: bool = True

class JapaneseTextQualityFilter:
    """日本語テキスト品質フィルター"""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
        
        # 不適切コンテンツパターン
        self.adult_patterns = [
            r'エロ', r'無修正', r'アダルト', r'エッチ', r'セックス',
            r'風俗', r'AV', r'官能', r'ラブホ', r'出会い系',
            r'セクシー', r'ヌード', r'巨乳', r'美乳', r'フェチ'
        ]
        
        # スパム・低品質パターン
        self.spam_patterns = [
            r'このサイトは', r'アクセス解析', r'SEO対策',
            r'クリックして', r'詳細はこちら', r'無料登録',
            r'会員登録', r'ログイン', r'パスワード',
            r'エラーが発生', r'ページが見つかりません',
            r'Javascript', r'Cookie', r'ブラウザ',
            r'メールアドレス', r'電話番号', r'住所',
            r'投稿日:', r'カテゴリー', r'タグ:'
        ]
        
        # 日本語文字パターン
        self.japanese_pattern = re.compile(r'[あ-んア-ン一-龯ー]')
        
        # 重複検出用セット
        self.seen_texts: Set[str] = set()
    
    def is_high_quality_japanese(self, text: str) -> bool:
        """高品質日本語テキストかどうかの判定"""
        
        # 基本的な長さチェック
        if len(text) < self.config.min_text_length or len(text) > self.config.max_text_length:
            return False
        
        # Unicode正規化
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # 日本語文字比率チェック
        japanese_chars = len(self.japanese_pattern.findall(text))
        total_chars = len(text)
        if total_chars > 0:
            japanese_ratio = japanese_chars / total_chars
            if japanese_ratio < self.config.min_japanese_ratio:
                return False
        
        # 不適切コンテンツフィルタリング
        if self.config.filter_adult_content:
            text_lower = text.lower()
            for pattern in self.adult_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False
        
        # スパムフィルタリング
        if self.config.filter_spam:
            for pattern in self.spam_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False
        
        # 重複チェック
        if self.config.remove_duplicates:
            text_signature = self._create_text_signature(text)
            if text_signature in self.seen_texts:
                return False
            self.seen_texts.add(text_signature)
        
        # HTML/XMLタグの除去チェック
        if self._contains_excessive_markup(text):
            return False
        
        # 連続する記号・空白のチェック
        if self._contains_excessive_symbols(text):
            return False
        
        return True
    
    def _create_text_signature(self, text: str) -> str:
        """テキストの署名（重複検出用）"""
        # 空白・記号を除去して最初の100文字
        cleaned = re.sub(r'[\s\W]', '', text)
        return cleaned[:100]
    
    def _contains_excessive_markup(self, text: str) -> bool:
        """過度なマークアップが含まれているかチェック"""
        # HTMLタグの数をチェック
        html_tags = len(re.findall(r'<[^>]+>', text))
        if html_tags > 5:  # 5個以上のHTMLタグがある場合
            return True
        
        # URL数のチェック
        urls = len(re.findall(r'https?://[^\s]+', text))
        if urls > 3:  # 3個以上のURLがある場合
            return True
        
        return False
    
    def _contains_excessive_symbols(self, text: str) -> bool:
        """過度な記号・空白が含まれているかチェック"""
        # 連続する記号
        if re.search(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?~`]{10,}', text):
            return True
        
        # 連続する空白
        if re.search(r'\s{10,}', text):
            return True
        
        # 同じ文字の連続
        if re.search(r'(.)\1{20,}', text):
            return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        # Unicode正規化
        text = unicodedata.normalize('NFKC', text)
        
        # HTMLタグの除去
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL除去
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # 連続空白の正規化
        text = re.sub(r'\s+', ' ', text)
        
        # 前後の空白除去
        text = text.strip()
        
        return text

class DatasetFormatter:
    """データセット整形クラス"""
    
    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.quality_filter = JapaneseTextQualityFilter(config)
    
    def format_existing_datasets(self, dataset_dir: Path) -> Dict[str, int]:
        """既存データセットの整形"""
        logger.info(f"Formatting datasets in: {dataset_dir}")
        
        results = {}
        
        # 対象ファイル
        target_files = [
            "wikipedia_ja.jsonl",
            "cc100_ja.jsonl",
            "oscar_ja.jsonl",
            "aozora_bunko.jsonl",
            "conversation_ja.jsonl"
        ]
        
        for filename in target_files:
            filepath = dataset_dir / filename
            if filepath.exists():
                logger.info(f"Processing: {filename}")
                processed_count = self._format_jsonl_file(filepath)
                results[filename] = processed_count
                logger.info(f"Processed {processed_count} high-quality entries from {filename}")
            else:
                logger.warning(f"File not found: {filename}")
                results[filename] = 0
        
        return results
    
    def _format_jsonl_file(self, filepath: Path) -> int:
        """JSONLファイルの整形"""
        high_quality_data = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        
                        if not text:
                            continue
                        
                        # テキストクリーニング
                        cleaned_text = self.quality_filter.clean_text(text)
                        
                        # 品質チェック
                        if self.quality_filter.is_high_quality_japanese(cleaned_text):
                            # データ構造の標準化
                            standardized_data = {
                                "text": cleaned_text,
                                "source": data.get('source', filepath.stem),
                                "quality_score": self._calculate_quality_score(cleaned_text)
                            }
                            
                            # 追加メタデータがあれば保持
                            if 'title' in data:
                                standardized_data['title'] = data['title']
                            
                            high_quality_data.append(standardized_data)
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON at line {line_num} in {filepath}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {filepath}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return 0
        
        # 品質順でソート
        high_quality_data.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # バックアップ作成
        backup_path = filepath.with_suffix('.bak')
        if not backup_path.exists():
            filepath.rename(backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # 整形済みデータの書き込み
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in high_quality_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Formatted {filepath}: {len(high_quality_data)} high-quality entries")
        return len(high_quality_data)
    
    def _calculate_quality_score(self, text: str) -> float:
        """テキストの品質スコア計算"""
        score = 0.0
        
        # 長さスコア（適度な長さを評価）
        length = len(text)
        if 50 <= length <= 500:
            score += 0.3
        elif 500 < length <= 1000:
            score += 0.2
        elif 20 <= length < 50:
            score += 0.1
        
        # 日本語文字比率
        japanese_chars = len(re.findall(r'[あ-んア-ン一-龯ー]', text))
        if length > 0:
            japanese_ratio = japanese_chars / length
            score += japanese_ratio * 0.4
        
        # 句読点の適切な使用
        punctuation_count = len(re.findall(r'[。、！？]', text))
        sentences = len(re.split(r'[。！？]', text))
        if sentences > 1:
            punctuation_ratio = punctuation_count / sentences
            score += min(punctuation_ratio, 1.0) * 0.2
        
        # 語彙の多様性（簡易版）
        words = set(re.findall(r'[あ-んア-ン一-龯ー]+', text))
        if len(text) > 0:
            vocabulary_diversity = len(words) / max(len(text) // 10, 1)
            score += min(vocabulary_diversity, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def create_validation_dataset(self, dataset_dir: Path, validation_ratio: float = 0.1) -> str:
        """検証用データセット作成"""
        logger.info("Creating validation dataset...")
        
        all_high_quality_data = []
        
        # 全データセットから高品質データを収集
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            if jsonl_file.name.startswith("validation_"):
                continue
            
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if data.get('quality_score', 0) > 0.5:  # 高品質のみ
                                all_high_quality_data.append(data)
                        except:
                            continue
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")
        
        if not all_high_quality_data:
            logger.warning("No high-quality data found for validation dataset")
            return ""
        
        # ランダムサンプリング
        random.shuffle(all_high_quality_data)
        validation_size = int(len(all_high_quality_data) * validation_ratio)
        validation_data = all_high_quality_data[:validation_size]
        
        # 検証データセット保存
        validation_file = dataset_dir / "validation_ja.jsonl"
        with open(validation_file, 'w', encoding='utf-8') as f:
            for entry in validation_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Validation dataset created: {validation_file} ({len(validation_data)} entries)")
        return str(validation_file)

def main():
    """メイン処理"""
    # 設定
    config = DataQualityConfig(
        min_text_length=20,
        max_text_length=1500,
        min_japanese_ratio=0.4,
        filter_adult_content=True,
        filter_spam=True,
        normalize_unicode=True,
        remove_duplicates=True
    )
    
    # データセットディレクトリ
    dataset_dir = Path("dataset/deepseek-jp")
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    
    # データセット整形実行
    formatter = DatasetFormatter(config)
    
    print("=" * 60)
    print("データセット品質向上・整形処理を開始します")
    print("=" * 60)
    
    # 既存データセットの整形
    results = formatter.format_existing_datasets(dataset_dir)
    
    # 結果表示
    print("\n" + "=" * 60)
    print("整形結果:")
    total_entries = 0
    for filename, count in results.items():
        print(f"  {filename}: {count} 高品質エントリ")
        total_entries += count
    
    print(f"  総計: {total_entries} 高品質エントリ")
    
    # 検証データセット作成
    validation_file = formatter.create_validation_dataset(dataset_dir)
    if validation_file:
        print(f"  検証データセット: {Path(validation_file).name}")
    
    print("=" * 60)
    print("データセット品質向上完了！")
    print("学習に適した高品質な日本語データセットが準備されました。")
    print("=" * 60)

if __name__ == "__main__":
    main()
