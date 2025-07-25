# DeepSeek R1 Analysis and Evaluation System

Author: Akira Ito a.k.a limonene213u  
DeepSeek R1の分析と評価のための包括的システム

## システム構成

### 1. 分析・評価システム (analyze_deepseekr1.py / analyze_deepseekr1_lite.py)

DeepSeek R1モデルの詳細分析を行うシステムです。

#### 主要機能
- **トークナイザー分析**: 語彙サイズ、特殊トークン、日本語対応状況
- **アーキテクチャ分析**: レイヤー構成、パラメータ数、MoE構造
- **日本語性能評価**: サンプルテキストでの処理能力測定
- **ベンチマーク実行**: 推論速度、メモリ使用量の測定

#### 使用方法
```bash
# 完全版（全依存関係が利用可能な環境）
python analyze_deepseekr1.py --model deepseek-ai/deepseek-r1-distill-qwen-1.5b

# 軽量版（制限された環境用）
python analyze_deepseekr1_lite.py --model deepseek-ai/deepseek-r1-distill-qwen-1.5b
```

### 2. 評価システム (evaluation/)

#### jlce_benchmark.py
**目的**: JLCE評価システムの実行とベンチマーク測定

**主要機能**:
- 包括的JLCE評価実行
- JGLUE標準ベンチマーク
- 処理速度ベンチマーク
- 非同期評価処理

**使用方法**:
```bash
# 包括的評価
python evaluation/jlce_benchmark.py --benchmark comprehensive --models deepseek-ai/deepseek-r1-distill-qwen-1.5b

# JGLUE評価
python evaluation/jlce_benchmark.py --benchmark jglue --models model1 model2

# 速度ベンチマーク
python evaluation/jlce_benchmark.py --benchmark speed --models deepseek-ai/deepseek-r1-distill-qwen-1.5b
```

#### comparative_analysis.py
**目的**: モデル間性能比較と統計的有意性検定

**主要機能**:
- モデル間比較分析
- 統計的有意性検定（scipy使用）
- Cohen's d効果量計算
- モデルランキング生成

**使用方法**:
```bash
# 評価結果の比較分析
python evaluation/comparative_analysis.py --results-dir evaluation_results --baseline model1

# 特定パターンのファイルを対象とした分析
python evaluation/comparative_analysis.py --pattern "*benchmark*.json"
```

#### performance_metrics.py
**目的**: 統計解析と性能測定ユーティリティ

**主要機能**:
- 実行時間・メモリ使用量測定
- スループット指標計算
- トークン化速度ベンチマーク
- 統計サマリー生成

**使用方法**:
```bash
# デモベンチマーク実行
python evaluation/performance_metrics.py --demo

# カスタム出力ファイル指定
python evaluation/performance_metrics.py --demo --output custom_metrics.json
```

### 3. 分析システム (analysis/)
将来的な分析機能拡張用ディレクトリ

### 4. ベンチマークシステム (benchmarking/)
将来的なベンチマーク機能拡張用ディレクトリ

## 依存関係

### 必須パッケージ
- torch
- transformers  
- numpy
- pandas
- pathlib
- json
- asyncio

### オプションパッケージ
- scipy (統計検定用)
- psutil (システム監視用)
- matplotlib (可視化用)

### インストール方法
```bash
# 基本パッケージ
pip install torch transformers numpy pandas

# 統計分析用
pip install scipy

# システム監視用  
pip install psutil

# 可視化用
pip install matplotlib
```

## 出力形式

### 評価結果
評価結果は `evaluation_results/` ディレクトリに保存されます：

- `comprehensive_evaluation_YYYYMMDD_HHMMSS.json`: 包括的評価結果
- `jglue_benchmark_YYYYMMDD_HHMMSS.json`: JGLUE評価結果
- `speed_benchmark_YYYYMMDD_HHMMSS.json`: 速度ベンチマーク結果
- `model_comparison_YYYYMMDD_HHMMSS.json`: モデル比較結果
- `model_comparison_YYYYMMDD_HHMMSS.md`: 比較レポート

### 分析結果
分析結果は以下の形式で出力されます：

- JSON形式: 機械可読な詳細データ
- Markdown形式: 人間可読なレポート
- 統計サマリー: 平均値、標準偏差、信頼区間等

## 注意事項

### 相対パス対応
評価システムのスクリプトは、ディレクトリ階層変更に対応するため、以下のパス設定を使用しています：

```python
sys.path.append(str(Path(__file__).parent.parent.parent))
```

### エラーハンドリング
- 依存関係が不足している場合、基本機能は維持される
- オプション機能が利用できない場合、警告を表示して継続実行
- 評価エラーが発生した場合、エラー情報を記録して次の処理に継続

### パフォーマンス考慮
- 大きなモデルでの評価時は十分なメモリを確保
- 並列処理を活用したい場合は、適切なワーカー数を設定
- 長時間実行される評価の場合、中間結果の保存を推奨

## トラブルシューティング

### よくある問題

1. **ImportError**: 必要なモジュールがインストールされていない
   - 解決方法: 上記の依存関係セクションに従ってパッケージをインストール

2. **ModuleNotFoundError**: 相対パスでモジュールが見つからない
   - 解決方法: ROCm-DeepSeek_R1-ja/Python/Analyze_DeepSeekR1/ から実行

3. **CUDA/ROCm関連エラー**: GPU環境設定の問題
   - 解決方法: CPU環境での実行を試すか、環境設定を確認

### サポート
問題が発生した場合は、以下の情報を含めて報告してください：
- 実行環境（OS、Python版、GPU情報）
- エラーメッセージの全文
- 実行したコマンド
- 関連する設定ファイル

---
*最終更新: 2025年7月25日*
