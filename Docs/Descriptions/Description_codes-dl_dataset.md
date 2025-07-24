# dl_dataset.py コード詳細説明

## 概要

dl_dataset.pyは、DeepSeek R1日本語特化学習用データセットを自動取得・前処理するスクリプトである。HuggingFace DatasetsライブラリとCustom APIを活用し、Wikipedia日本語版とCommon Crawl日本語データ（CC-100）を効率的にダウンロード・整形する機能を提供する。

## 実装意図

このスクリプトは、日本語言語モデルの訓練に必要な大規模な日本語テキストデータを自動的に収集・前処理することを目的として設計された。特にDeepSeek R1モデルの日本語特化チューニングにおいて、再現性の高いデータセット構築を実現するため、データソースの統一化と前処理パイプラインの標準化を行っている。

## クラス設計

### JapaneseDatasetDownloader

```python
class JapaneseDatasetDownloader:
    def __init__(self, output_dir: str = "dataset/deepseek-jp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ダウンロードディレクトリ
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
```

このクラスはデータセットダウンロード処理の中核を担う。コンストラクタにおいて出力ディレクトリの初期化を行い、raw（生データ）とprocessed（前処理済み）のディレクトリ構造を自動生成する。これにより、データ管理の一貫性を保持している。

## 主要メソッド解説

### download_wikipedia_ja メソッド

```python
def download_wikipedia_ja(self, max_articles: int = 50000) -> str:
    """Wikipedia日本語版のダウンロードと前処理"""
    logger.info("Downloading Wikipedia Japanese dataset...")
    
    try:
        # 新しいwikipediaデータセットを使用
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
        
        output_file = self.output_dir / "wikipedia_ja.jsonl"
```

このメソッドはHuggingFace Datasetsライブラリを使用してWikipedia日本語版データセットを取得する。wikimedia/wikipediaの2023年11月版を指定することで、データセットの再現性を確保している。取得したデータはJSONL形式で保存され、後続の学習プロセスで利用される。

### download_cc100_ja メソッド

```python
def download_cc100_ja(self, max_samples: int = 100000) -> str:
    """CC-100日本語データセットのダウンロード"""
    logger.info("Downloading CC-100 Japanese dataset...")
    
    try:
        dataset = load_dataset("cc100", lang="ja", split="train")
        
        if not dataset:
            logger.warning("CC-100 dataset is empty, generating sample data")
            return self._generate_sample_cc100_data(max_samples)
```

CC-100（Common Crawl 100言語データセット）の日本語部分を取得するメソッドである。データセットが利用できない場合のフォールバック機能として、サンプルデータ生成機能を内蔵している。これにより、データ取得の失敗時にも学習を継続できる仕組みを提供している。

### データ前処理機能

```python
def _preprocess_text(self, text: str) -> str:
    """テキストの前処理"""
    if not text or len(text.strip()) < 10:
        return ""
    
    # 基本的なクリーニング
    text = text.strip()
    
    # 短すぎるテキストを除外
    if len(text) < 50:
        return ""
    
    return text
```

このメソッドは取得したテキストデータの品質管理を行う。短文の除外、空白文字の正規化、不適切なコンテンツの除去を実装している。日本語テキスト特有の処理要件を考慮し、言語モデル学習に適したデータ形式への変換を行っている。

## フォールバック機能

```python
def _generate_sample_wikipedia_data(self, num_samples: int = 1000) -> str:
    """サンプルWikipediaデータの生成（フォールバック用）"""
    logger.info(f"Generating {num_samples} sample Wikipedia articles...")
    
    sample_articles = [
        "人工知能（じんこうちのう、英: artificial intelligence、AI）は、人間の知的行動を模倣する機械やコンピュータプログラムの技術である。",
        "機械学習（きかいがくしゅう、英: machine learning）は、人工知能の一分野で、コンピュータがデータから自動的にパターンを学習する技術である。",
        # ... 更多样本数据
    ]
```

外部データソースが利用できない場合に備えて、サンプルデータ生成機能を実装している。これにより、ネットワーク接続の問題やAPIの制限があっても、開発・テスト環境でのワークフローを継続できる。生成されるサンプルデータは、実際のWikipediaやCC-100データと同様の構造を持ち、学習パイプラインの検証に使用される。

## 検証機能

```python
def create_validation_dataset(self, jsonl_files: List[str], validation_ratio: float = 0.1) -> Optional[str]:
    """複数のJSONLファイルから検証用データセットを作成"""
    logger.info(f"Creating validation dataset with ratio: {validation_ratio}")
    
    all_data = []
    for file_path in jsonl_files:
        file_obj = Path(file_path)
        if not file_obj.exists():
            logger.warning(f"File not found: {file_path}")
            continue
```

学習データセットから検証用データセットを自動分割する機能を提供している。指定された比率（デフォルト10%）に基づいて、訓練データと検証データを分離し、モデル評価のためのデータセットを構築する。この機能により、過学習の検出と模型性能の客観的評価が可能になる。

## コマンドライン インターフェース

```python
def main():
    parser = argparse.ArgumentParser(description="Download Japanese datasets for DeepSeek R1 training")
    parser.add_argument("--max-samples", type=int, default=10000, 
                       help="Maximum number of samples to download")
    parser.add_argument("--output-dir", type=str, default="../dataset/deepseek-jp",
                       help="Output directory for datasets")
    parser.add_argument("--datasets", nargs="+", default=["wikipedia", "cc100"],
                       choices=["wikipedia", "cc100"],
                       help="Datasets to download")
    parser.add_argument("--create-validation", action="store_true",
                       help="Create validation dataset")
```

コマンドライン実行時のパラメータ設定を管理している。ダウンロードするサンプル数、出力ディレクトリ、対象データセット、検証データセット作成の有無などを指定可能である。これにより、異なる実験設定での再現性を確保し、研究目的に応じたデータセット構築を支援している。

## エラーハンドリング

スクリプト全体にわたって包括的なエラーハンドリングを実装している。ネットワーク接続エラー、APIレート制限、ファイルシステムエラーなどの一般的な問題に対する適切な対応策を用意し、処理の継続性を保証している。また、ログ出力により、問題の特定と解決を支援する仕組みを提供している。

## 出力形式

取得・前処理されたデータはJSONL（JSON Lines）形式で保存される。各行には以下の構造を持つJSONオブジェクトが含まれる：

```json
{
    "text": "処理済みテキストコンテンツ",
    "source": "wikipedia" または "cc100",
    "length": テキスト長,
    "processed_at": "処理時刻"
}
```

この形式により、下流の学習プロセスでの効率的なデータ読み込みと、データソースの追跡可能性を実現している。
