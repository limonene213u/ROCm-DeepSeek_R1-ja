# DeepSeek R1 Japanese Language Adaptation: Statistical Analysis Framework

## Overview

This R package implements the JLCE (Japanese Language Comprehension Evaluation) mathematical framework described in the research paper. It provides comprehensive statistical analysis tools for evaluating Japanese language model performance across multiple dimensions.

## Mathematical Foundation

The framework implements the core JLCE equation:

```
JLCE_Score = α·P(semantic) + β·P(syntactic) + γ·P(pragmatic) + δ·C(cultural)
```

Where:
- **α=0.35, β=0.25, γ=0.25, δ=0.15** (empirically validated weights for Japanese evaluation)
- **P(semantic)**: Semantic accuracy probability through cross-lingual similarity
- **P(syntactic)**: Syntactic correctness via dependency parsing validation
- **P(pragmatic)**: Pragmatic appropriateness using contextual coherence
- **C(cultural)**: Cultural competency coefficient for Japanese-specific nuances

## File Structure & Responsibilities

### Core Analysis Framework

#### 📊 `analyze_deeepseekr1.r` - **JLCE Mathematical Framework Core**
**Primary Role**: 論文記載のJLCE数学的フレームワークの完全実装

**Key Functions**:
- `calculate_jlce_score()` - 4次元複合スコア計算
- `calculate_semantic_accuracy()` - 意味論的精度測定
- `bootstrap_jlce_ci()` - Bootstrap信頼区間算出
- `comparative_analysis()` - モデル間比較・効果量評価
- `run_jlce_analysis()` - 完全自動分析パイプライン

**Mathematical Implementation**:
```
JLCE_Score = α·P(semantic) + β·P(syntactic) + γ·P(pragmatic) + δ·C(cultural)
α=0.35, β=0.25, γ=0.25, δ=0.15 (empirically validated for Japanese)
```

**Usage**: 日本語言語モデル評価の標準ツール

---

#### 📈 `deepseek_r1_statistical_analysis.R` - **Performance Claims Validator**
**Primary Role**: 論文記載の効率性主張の統計的妥当性検証

**Validation Targets**:
- **MLA効率性**: KVキャッシュ削減率（5-13%）の仮説検定
- **LoRA効率性**: パラメータ削減（200倍）の信頼区間推定  
- **メモリ効率性**: メモリ削減（2倍）の対応t検定
- **処理速度**: 高速化（10.47倍, 7.60倍）のBootstrap検証

**Statistical Methods**:
- ベイジアン推定 + 頻度論的検定
- 多重比較補正（Bonferroni）
- 効果量評価（Cohen's d）

**Usage**: 学術的再現性確保・査読対応

---

#### 🧪 `example_usage.r` - **Comprehensive Tutorial & Testing Suite**
**Primary Role**: 実践的使用方法とテストフレームワーク

**Tutorial Content**:
- ステップバイステップ実行例
- 合成データ生成・テスト実行
- 統計結果の解釈方法
- 可視化出力の確認手順

**Testing Features**:
- 全機能の動作確認
- エラーハンドリング検証
- パフォーマンステスト

**Usage**: 初心者向け学習・開発者向けテスト

---

### Recommended Workflow

1. **学習・テスト**: `example_usage.r` でフレームワーク理解
2. **実際の評価**: `analyze_deeepseekr1.r` で日本語モデル分析
3. **妥当性検証**: `deepseek_r1_statistical_analysis.R` で統計的確認

## Features

### Core Statistical Functions

1. **`calculate_jlce_score()`**
   - Implements the mathematical JLCE composite scoring
   - Validates input ranges and dimensions
   - Returns interpretable [0,1] scores

2. **`calculate_semantic_accuracy()`**
   - Cosine similarity-based semantic evaluation
   - 768-dimensional embedding support
   - Configurable threshold (τ=0.65 for Japanese)

3. **`bootstrap_jlce_ci()`**
   - Bootstrap confidence intervals (B=1000 iterations)
   - Robust statistical inference
   - 95% confidence level by default

4. **`comparative_analysis()`**
   - Paired t-test with Bonferroni correction
   - Cohen's d effect size calculation
   - Statistical and practical significance testing

### Data Processing Functions

5. **`load_jlce_results()`**
   - JSON data loading and validation
   - Automatic JLCE score computation
   - Interpretive category assignment

6. **`load_prompt_metadata()`**
   - 31-prompt dataset processing
   - Linguistic complexity scoring
   - Domain-weighted analysis

### Visualization Tools

7. **`create_jlce_dashboard()`**
   - Four-dimensional performance analysis
   - Composite score distribution
   - Competency category breakdown

8. **`plot_bootstrap_ci()`**
   - Bootstrap distribution visualization
   - Confidence interval display
   - Statistical summary plots

### Analysis Pipeline

9. **`run_jlce_analysis()`**
   - Complete end-to-end analysis
   - Automated report generation
   - High-quality plot export

## Installation and Setup

### Required R Packages

```r
# Install required packages
install.packages(c(
  "dplyr", "ggplot2", "readr", "jsonlite", 
  "boot", "effsize", "binom", "reshape2", 
  "gridExtra", "viridis"
))
```

### Loading the Framework

```r
# Load the analysis framework
source("analyze_deeepseekr1.r")
```

## Usage Examples

### Basic JLCE Score Calculation

```r
# Generate sample data
sample_data <- generate_synthetic_jlce_data(31)

# Calculate composite JLCE scores
jlce_scores <- calculate_jlce_score(
  semantic_scores = sample_data$semantic,
  syntactic_scores = sample_data$syntactic,
  pragmatic_scores = sample_data$pragmatic,
  cultural_scores = sample_data$cultural
)

print(paste("Mean JLCE Score:", round(mean(jlce_scores), 3)))
```

### Bootstrap Confidence Intervals

```r
# Calculate 95% confidence intervals
bootstrap_results <- bootstrap_jlce_ci(jlce_scores, B = 1000)

cat("95% CI:", round(bootstrap_results$ci_lower, 4), 
    "to", round(bootstrap_results$ci_upper, 4))
```

### Model Comparison

```r
# Compare baseline vs optimized models
comparison <- comparative_analysis(baseline_scores, optimized_scores)

cat("Effect Size (Cohen's d):", round(comparison$cohen_d, 4))
cat("Interpretation:", comparison$effect_size_interpretation)
```

### Complete Analysis Pipeline

```r
# Run comprehensive analysis
results <- run_jlce_analysis(
  results_file = "jlce_evaluation_results.json",
  prompts_file = "prompts_swallow_bench.jsonl",
  output_dir = "analysis_output"
)
```

## Data Format Requirements

### JLCE Results JSON Format

```json
{
  "semantic_scores": [0.85, 0.78, 0.92, ...],
  "syntactic_scores": [0.82, 0.75, 0.89, ...],
  "pragmatic_scores": [0.79, 0.73, 0.86, ...],
  "cultural_scores": [0.77, 0.71, 0.84, ...],
  "metadata": {
    "model_name": "DeepSeek-R1-Japanese-Adapter",
    "evaluation_date": "2025-07-29",
    "total_prompts": 31
  }
}
```

### Prompts JSONL Format

```jsonl
{"prompt_id": 1, "domain": "Technical", "prompt_text": "...", "expected_output": "..."}
{"prompt_id": 2, "domain": "Social Policy", "prompt_text": "...", "expected_output": "..."}
```

## Output Files

The analysis pipeline generates:

1. **`jlce_dashboard.png`** - Comprehensive performance visualization
2. **`bootstrap_ci.png`** - Bootstrap confidence interval plot
3. **`jlce_summary_statistics.json`** - Statistical summary report

## Interpretation Guidelines

### JLCE Score Ranges (0-100 scale)

- **90-100**: Native-level Japanese competency
- **80-89**: Advanced Japanese understanding
- **70-79**: Intermediate Japanese capability
- **60-69**: Basic Japanese comprehension
- **<60**: Limited Japanese functionality

### Effect Size Interpretation (Cohen's d)

- **|d| > 0.8**: Large effect (substantial improvement)
- **|d| > 0.5**: Medium effect (moderate improvement)
- **|d| > 0.2**: Small effect (minimal improvement)
- **|d| ≤ 0.2**: Negligible effect

## Example Workflow

1. **Data Preparation**: Format evaluation results as JSON
2. **Load Framework**: `source("analyze_deeepseekr1.r")`
3. **Run Analysis**: `run_jlce_analysis(results_file, prompts_file)`
4. **Review Outputs**: Check generated plots and statistics
5. **Statistical Interpretation**: Use provided guidelines

## Academic Citation

If you use this framework in your research, please cite:

```bibtex
@article{ito2025deepseek,
  title={DeepSeek R1 Japanese Language Adaptation: A Comprehensive Implementation and Validation Framework},
  author={Ito, Akira},
  journal={AETS Technical Report},
  year={2025},
  url={https://github.com/limonene213u/ROCm-DeepSeek_R1-ja}
}
```

## License

BSD-3-Clause License - See LICENSE file for details.

## Support and Contributions

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Author**: Akira Ito (AETS - Akatsuki Enterprise Technology Solutions)  
**Date**: 2025-07-29  
**Version**: 1.0.0
