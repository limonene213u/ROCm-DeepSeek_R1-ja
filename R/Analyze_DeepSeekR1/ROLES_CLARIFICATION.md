# R統計分析フレームワーク - 役割分担明確化

## 📊 ファイル構成と責任範囲

### コアファイル詳細

---

## 1️⃣ `analyze_deeepseekr1.r` - **JLCE数学的フレームワーク実装**

### 🎯 **Primary Role**
論文記載のJLCE（Japanese Language Comprehension Evaluation）数学的フレームワークの完全実装

### 📐 **Mathematical Foundation**
```r
# Core JLCE equation implementation
JLCE_Score = α·P(semantic) + β·P(syntactic) + γ·P(pragmatic) + δ·C(cultural)

# Empirically validated weights for Japanese evaluation
α=0.35  # Semantic weight (highest priority)
β=0.25  # Syntactic weight
γ=0.25  # Pragmatic weight  
δ=0.15  # Cultural weight
```

### 🔧 **Core Functions**
- `calculate_jlce_score()` - 4次元複合スコア計算
- `calculate_semantic_accuracy()` - コサイン類似度による意味論的精度測定
- `bootstrap_jlce_ci()` - Bootstrap信頼区間算出（B=1000）
- `comparative_analysis()` - 対応t検定 + Cohen's d効果量
- `run_jlce_analysis()` - 完全自動分析パイプライン

### 🎯 **Target Users**
- 日本語言語モデル評価研究者
- AI開発者・エンジニア
- 学術研究者（論文執筆・査読）

---

## 2️⃣ `deepseek_r1_statistical_analysis.R` - **効率性主張統計検証**

### 🎯 **Primary Role**
論文記載の効率性向上主張に対する統計的妥当性検証

### 📊 **Validation Targets**
```r
# Paper claims requiring statistical verification
MLA_efficiency:    "5-13% KV cache reduction"     # Hypothesis testing
LoRA_parameters:   "200x parameter reduction"     # Confidence intervals
LoRA_memory:       "2x memory reduction"          # Paired t-test
Speed_improvements: "10.47x, 7.60x faster"       # Bootstrap validation
```

### 🔬 **Statistical Methods**
- **ベイジアン推定**: 事前分布設定による堅牢な推定
- **頻度論的検定**: 仮説検定による客観的判定
- **多重比較補正**: Bonferroni補正による厳密性確保
- **効果量評価**: Cohen's dによる実用的有意性測定

### 🎯 **Target Users**
- 統計学者・データサイエンティスト
- 査読者・編集者
- 再現性検証研究者

---

## 3️⃣ `example_usage.r` - **包括的チュートリアル・テストスイート**

### 🎯 **Primary Role**
フレームワークの実践的使用方法詳細説明と動作確認

### 📚 **Tutorial Content**
- **基本使用例**: JLCE スコア計算からBootstrap信頼区間まで
- **モデル比較**: ベースライン vs 最適化モデルの統計的比較
- **可視化例**: ダッシュボード作成・Bootstrap分布プロット
- **完全パイプライン**: JSON/JSONL データ処理の実例

### 🧪 **Testing Framework**
- **合成データ生成**: `generate_synthetic_jlce_data()` による再現可能テスト
- **エラーハンドリング**: 各種異常ケースの動作確認
- **パフォーマンステスト**: 大規模データでの処理時間測定

### 🎯 **Target Users**
- 初心者・学習者（フレームワーク理解）
- 開発者（機能テスト・デバッグ）
- 教育者（統計分析教材）

---

## 🔄 **推奨ワークフロー**

### 段階1: 学習・理解
```bash
# チュートリアル実行でフレームワーク理解
Rscript example_usage.r
```

### 段階2: 実際の評価
```r
# 日本語モデルの実際のJLCE評価
source("analyze_deeepseekr1.r")
results <- run_jlce_analysis("evaluation_results.json", "prompts.jsonl")
```

### 段階3: 妥当性検証
```r
# 論文主張の統計的妥当性確認
source("deepseek_r1_statistical_analysis.R")
validation <- main_analysis()
```

---

## 📈 **統計的厳密性レベル**

### Level 1: 基本分析 (`analyze_deeepseekr1.r`)
- 記述統計・信頼区間
- 標準的な仮説検定
- 効果量計算

### Level 2: 高度検証 (`deepseek_r1_statistical_analysis.R`)  
- ベイジアン分析
- 多重比較補正
- ロバスト推定

### Level 3: 完全検証 (Combined Analysis)
- クロスバリデーション
- 感度分析
- 学術的品質保証

---

## 🎓 **学術的貢献**

### 論文品質向上
- **再現性**: 全実装コードの公開
- **透明性**: 統計手法の明示
- **厳密性**: 多層的検証アプローチ

### 研究インパクト
- **標準化**: 日本語LM評価の統一基準
- **自動化**: 評価プロセスの効率化
- **教育**: 統計分析のベストプラクティス提供

---

**最終更新**: 2025-07-29  
**バージョン**: 1.0.0  
**Total Implementation**: 1,417行の包括的統計分析フレームワーク
