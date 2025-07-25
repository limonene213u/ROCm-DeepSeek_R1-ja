# DeepSeek R1 BPE解析ツール

DeepSeek R1 Distillシリーズのトークナイザーにおける日本語対応状況を分析するツールです。

## ファイル構成

- `analyze_deepseekr1.py` - フル機能版（可視化付き）
- `analyze_deepseekr1_lite.py` - 軽量版（可視化なし）
- `requirements.txt` - フル機能版の依存関係
- `requirements-lite.txt` - 軽量版の依存関係

## セットアップ

### 軽量版（推奨）

```bash
pip install -r requirements-lite.txt
```

### フル機能版（可視化付き）

```bash
pip install -r requirements.txt
```

## 使用方法

### 軽量版の実行

```bash
python analyze_deepseekr1_lite.py
```

### フル機能版の実行

```bash
python analyze_deepseekr1.py
```

## 解析対象モデル

1. **DeepSeek-R1-Distill-Llama-8B** (16GB)
2. **DeepSeek-R1-Distill-Qwen-14B** (28GB)  
3. **DeepSeek-R1-Distill-Qwen-32B** (64GB)
4. **deepseek-r1-distill-qwen-1.5b** (4GB)

## 出力ファイル

解析結果は `Analyze_DeepSeekR1_Data/` ディレクトリに以下の形式で保存されます：

- `model_comparison.csv` - モデル比較データ
- `detailed_analysis_results.json` - 詳細分析結果
- `analysis_summary_report.md` - 分析サマリーレポート
- `tokenizer_analysis_visualization.png` - 可視化グラフ（フル機能版のみ）

## 分析内容

- 語彙サイズとトークン分布
- 日本語トークンの分類（ひらがな、カタカナ、漢字、混在）
- サブワード分割効率の測定
- 一般的な日本語単語のカバレッジ評価
- トークン長の統計分析

## システム要件

- Python 3.8以上
- 十分なメモリ（モデルサイズに依存）
- インターネット接続（初回モデルダウンロード時）

## 注意事項

- 初回実行時は各モデルのダウンロードに時間がかかります
- 大きなモデル（32B）の解析には相当なメモリが必要です
- ROCm環境でも動作しますが、CPU環境でも実行可能です
