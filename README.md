# ROCm-DeepSeek_R1-jp

## **プロジェクト構成**

```text
ROCm-DeepSeek_R1-ja/
├── Python/
│   ├── Adapters/           # モデル適応・学習
│   ├── Benchmark/          # 性能ベンチマーク
│   ├── DataProcessing/     # データセット処理
│   ├── ModelAnalysis/      # モデル解析
│   ├── Validation/         # 論文検証
│   └── setup/             # 環境セットアップ
├── R/                     # R統計分析
├── Docs/                  # ドキュメント
├── tests/                 # テストスイート
└── tools/                 # 補助ツール
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
python Python/Adapters/deepseek_ja_adapter.py --mode development
# 起動時にモデル選択画面が表示されます
```

### 2. モデルタイプで指定

```bash
# Llama 8Bモデル（軽量・高速）
python Python/Adapters/deepseek_ja_adapter.py --model-type llama-8b --mode production

# Qwen 14Bモデル（バランス型）
python Python/Adapters/deepseek_ja_adapter.py --model-type qwen-14b --mode production

# Qwen 32Bモデル（高性能）
python Python/Adapters/deepseek_ja_adapter.py --model-type qwen-32b --mode production
```

### 3. 直接モデル名で指定

```bash
python Python/Adapters/deepseek_ja_adapter.py --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --mode production
```

### 4. サポートモデル一覧表示

```bash
python Python/Adapters/deepseek_ja_adapter.py --show-models
```

### 5. 設定確認（ドライラン）

```bash
# 実際の学習を行わずに設定のみ確認
python Python/Adapters/deepseek_ja_adapter.py --model-type qwen-14b --mode production --dry-run
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
python Python/Adapters/deepseek_ja_adapter.py --mode trial --auto

# 開発モード（サンプル200件でテスト）
python Python/Adapters/deepseek_ja_adapter.py --mode development --epochs 3

# 本格運用（実データセット必須）
python Python/Adapters/deepseek_ja_adapter.py --mode production --epochs 10
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
python Python/Adapters/deepseek_ja_adapter.py --model-type qwen-32b --mode production --epochs 10
```

### 軽量モデルでの開発テスト

```bash
# Llama 8Bモデルで開発モード
python Python/Adapters/deepseek_ja_adapter.py --model-type llama-8b --mode development --epochs 3
```

### 継続学習

```bash
# 既存モデルから継続学習
python Python/Adapters/deepseek_ja_adapter.py --model-type qwen-14b --continue-from ./output/previous_model --epochs 5
```

## **ライセンス**

本プロジェクトは **BSD-3-Clause License** の下で公開されています。

```
BSD 3-Clause License

Copyright (c) 2025, Akira Ito - AETS (Akatsuki Enterprise Technology Solutions)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### 学術利用について

本プロジェクトは学術研究および教育目的での利用を主眼としています：

- **研究利用**: 論文執筆、学会発表での自由な利用を許可
- **再現性**: 全実装詳細とデータセットの公開による完全再現可能性
- **引用**: 学術利用の際は適切な引用をお願いします

### 引用方法

```bibtex
@software{ito2025_deepseek_r1_ja,
  author = {Ito, Akira},
  title = {DeepSeek R1 Japanese Language Adaptation: A Comprehensive Implementation and Validation Framework},
  year = {2025},
  url = {https://github.com/limonene213u/ROCm-DeepSeek_R1-ja},
  organization = {AETS (Akatsuki Enterprise Technology Solutions)}
}
```

## **コントリビュート**

プロジェクトへの貢献を歓迎します：

1. **Issue報告**: バグ報告や機能要望
2. **Pull Request**: コード改善や新機能追加
3. **文書化**: ドキュメント改善やチュートリアル追加
4. **テスト**: 新しい環境でのテスト結果報告

詳細は [CONTRIBUTING.md](CONTRIBUTING.md) をご参照ください。