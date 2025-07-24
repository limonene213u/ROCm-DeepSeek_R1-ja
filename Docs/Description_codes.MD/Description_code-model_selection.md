# DeepSeek R1 Distillモデル選択機能 設計書

## 概要

本プロジェクトでは、3つのDeepSeek R1 Distillモデルをサポートし、それぞれに最適化された学習戦略を自動適用する機能を提供しています。

## サポートされているモデル

### 1. DeepSeek-R1-Distill-Llama-8B
- **特徴**: 推論重視・軽量
- **メモリ要件**: 16GB
- **推奨用途**: 高速推論、リソース制約環境
- **最適化戦略**:
  - バッチサイズ: 4
  - 勾配累積: 4
  - 学習率: 1e-4
  - LoRA rank: 16, alpha: 32

### 2. DeepSeek-R1-Distill-Qwen-14B
- **特徴**: バランス型
- **メモリ要件**: 28GB  
- **推奨用途**: 一般的な用途、バランス重視
- **最適化戦略**:
  - バッチサイズ: 2
  - 勾配累積: 8
  - 学習率: 8e-5
  - LoRA rank: 32, alpha: 64

### 3. DeepSeek-R1-Distill-Qwen-32B
- **特徴**: 高性能・大容量
- **メモリ要件**: 64GB
- **推奨用途**: 高品質生成、研究用途
- **最適化戦略**:
  - バッチサイズ: 1
  - 勾配累積: 16
  - 学習率: 5e-5
  - LoRA rank: 64, alpha: 128

## 実装アーキテクチャ

### クラス構成

#### SupportedModel (Enum)
```python
class SupportedModel(Enum):
    LLAMA_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    QWEN_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    QWEN_32B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    QWEN_1_5B = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"  # 後方互換性
```

#### ModelStrategy (dataclass)
```python
@dataclass
class ModelStrategy:
    model_name: str
    recommended_batch_size: int
    recommended_gradient_accumulation: int
    recommended_learning_rate: float
    recommended_lora_r: int
    recommended_lora_alpha: int
    memory_requirements_gb: float
    vram_optimized: bool = False
```

#### ModelRegistry
- 各モデルの最適戦略を管理
- インタラクティブな選択機能
- 戦略の取得とモデル一覧表示

### 選択方法

1. **対話形式選択**: `ModelRegistry.interactive_model_selection()`
2. **コマンドライン指定**: `--model-type`、`--model-name`
3. **自動選択**: `--auto`モードで最軽量モデル

### 学習戦略の適用

- **バッチサイズ**: モデルサイズに応じた最適値
- **学習率**: モデル別推奨値の自動設定
- **LoRA設定**: モデルサイズに応じたrank/alpha調整
- **シーケンス長**: メモリ要件に基づく動的調整

## 使用例

### 基本的な使用
```bash
# 対話形式でモデル選択
python Python/deepseek_ja_adapter.py --mode development

# モデルタイプ指定
python Python/deepseek_ja_adapter.py --model-type qwen-14b --mode production

# 直接指定
python Python/deepseek_ja_adapter.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

### 最適化された学習設定

各モデルに対して以下が自動適用されます：

- **Memory-Optimized Training**: メモリ要件に応じたバッチサイズ調整
- **Gradient Accumulation**: 効果的なバッチサイズの実現
- **LoRA Configuration**: モデルサイズに最適化されたパラメータ
- **Learning Rate Scheduling**: モデル別推奨学習率

## トレーナークラスの統合

`DeepSeekJapaneseTrainer`クラスに`ModelStrategy`を統合：

```python
def __init__(self, config: JapaneseDataConfig, model_strategy: Optional[ModelStrategy] = None):
    self.model_strategy = model_strategy or ModelRegistry.get_strategy(SupportedModel.QWEN_1_5B.value)
    self.batch_size = self.model_strategy.recommended_batch_size
    self.gradient_accumulation_steps = self.model_strategy.recommended_gradient_accumulation
```

## メリット

1. **最適化された学習**: モデル別最適パラメータの自動適用
2. **メモリ効率**: モデルサイズに応じたメモリ使用量調整
3. **ユーザビリティ**: 簡単なモデル選択インターフェース
4. **拡張性**: 新しいモデルの追加が容易
5. **一貫性**: 統一されたパフォーマンス設定

## 今後の拡張

- さらなるDeepSeekモデルのサポート
- 動的なハードウェア検出に基づく最適化
- カスタム戦略の定義機能
- 学習結果に基づく戦略の自動調整
