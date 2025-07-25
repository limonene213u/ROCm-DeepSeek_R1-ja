# ROCm-DeepSeek_R1-jp

## **セットアップ**

### 依存関係のインストール

#### 標準インストール（推奨）

```bash
pip install -r requirements.txt
```

#### 軽量版（可視化機能なし）

```bash
pip install -r requirements-lite.txt
```

#### ROCm環境での最適化インストール

```bash
# 標準依存関係をインストール
pip install -r requirements.txt

# ROCm最適化パッケージを追加
pip install -r requirements-rocm.txt
```

### ROCm環境での特別な設定

AMD GPU（MI300X等）を使用する場合は、ROCm専用PyTorchの使用を推奨：

```bash
# ROCm 5.6用PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# または requirements-rocm.txt の指示に従ってください
```

## **サポートされているモデル**

本プロジェクトでは、以下のDeepSeek R1 Distillモデルをサポートしています：

| モデル | 特徴 | メモリ要件 | 推奨用途 |
|--------|------|------------|----------|
| **DeepSeek-R1-Distill-Llama-8B** | 推論重視・軽量 | 16GB | 高速推論、リソース制約環境 |
| **DeepSeek-R1-Distill-Qwen-14B** | バランス型 | 28GB | 一般的な用途、バランス重視 |
| **DeepSeek-R1-Distill-Qwen-32B** | 高性能・大容量 | 64GB | 高品質生成、研究用途 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | テスト用 | 4GB | 開発・実験用 |

各モデルには最適化された学習戦略（学習率、バッチサイズ、LoRA設定）が自動適用されます。

## **モデル選択方法**

### 1. 対話形式で選択（推奨）

```bash
python Python/deepseek_ja_adapter.py --mode development
# 起動時にモデル選択画面が表示されます
```

### 2. モデルタイプで指定

```bash
# Llama 8Bモデル（軽量・高速）
python Python/deepseek_ja_adapter.py --model-type llama-8b --mode production

# Qwen 14Bモデル（バランス型）
python Python/deepseek_ja_adapter.py --model-type qwen-14b --mode production

# Qwen 32Bモデル（高性能）
python Python/deepseek_ja_adapter.py --model-type qwen-32b --mode production
```

### 3. 直接モデル名で指定

```bash
python Python/deepseek_ja_adapter.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --mode production
```

### 4. サポートモデル一覧表示

```bash
python Python/deepseek_ja_adapter.py --show-models
```

### 5. 設定確認（ドライラン）

```bash
# 実際の学習を行わずに設定のみ確認
python Python/deepseek_ja_adapter.py --model-type qwen-14b --mode production --dry-run
```

## **ExecutionMode & DatasetManager**

```python
class ExecutionMode(Enum):
    PRODUCTION = "production"    # 実際のデータセット必要
    DEVELOPMENT = "development"  # サンプル生成OK
    TRIAL = "trial"             # 最小サンプル
```

**スマートな判定システム：**

- **本格運用**: 実データセット必須、エラーで停止
- **開発モード**: サンプル自動生成（200件）
- **試行モード**: 最小サンプル（50件）

## **使い方**

```bash
# 試行モード（最小サンプルで動作確認）
python Python/deepseek_ja_adapter.py --mode trial --auto

# 開発モード（サンプル200件でテスト）
python Python/deepseek_ja_adapter.py --mode development --epochs 3

# 本格運用（実データセット必須）
python Python/deepseek_ja_adapter.py --mode production --epochs 10
```

**改善されたポイント：**

1. **条件分岐の自動化** - モード判定でデータセット生成を制御
2. **エラーハンドリング** - 本格運用時のデータセット不備を事前チェック
3. **メンテナンス性向上** - クラス分離でロジック整理
4. **ログ整理** - 英語統一、簡潔化
5. **モデル選択機能** - 3つのDeepSeek R1 Distillモデルをサポート、最適化された学習戦略を自動適用

## **使用例**

### 高性能モデルでの本格学習
```bash
# Qwen 32Bモデルで本格運用
python Python/deepseek_ja_adapter.py --model-type qwen-32b --mode production --epochs 10
```

### 軽量モデルでの開発テスト
```bash
# Llama 8Bモデルで開発モード
python Python/deepseek_ja_adapter.py --model-type llama-8b --mode development --epochs 3
```

### 継続学習
```bash
# 既存モデルから継続学習
python Python/deepseek_ja_adapter.py --model-type qwen-14b --continue-from ./output/previous_model --epochs 5
```