# =============================================================================
# JLCE Framework Usage Examples & Testing Suite
# 包括的使用例・テストフレームワーク
# 
# PURPOSE: JLCEフレームワークの実践的使用方法の詳細説明
# - ステップバイステップ使用例
# - 合成データ生成によるテスト実行
# - 統計分析結果の解釈方法
# - 可視化出力の確認手順
# 
# SCOPE: 初心者から上級者まで対応の完全チュートリアル
# TARGET: フレームワーク利用者・開発者・学習者
# FEATURE: 実行可能なコード例 + 詳細解説
# 
# Author: Akira Ito (AETS - Akatsuki Enterprise Technology Solutions)
# Date: 2025-07-29
# License: BSD-3-Clause
# =============================================================================

# Load the analysis framework
source("analyze_deeepseekr1.r")

# Example 1: Basic JLCE Score Calculation
# =======================================

# Generate sample evaluation data
n_samples <- 31
sample_data <- generate_synthetic_jlce_data(n_samples)

# Calculate JLCE composite scores
jlce_scores <- calculate_jlce_score(
  semantic_scores = sample_data$semantic,
  syntactic_scores = sample_data$syntactic,
  pragmatic_scores = sample_data$pragmatic,
  cultural_scores = sample_data$cultural
)

print("Sample JLCE Scores:")
print(head(jlce_scores))
print(paste("Mean JLCE Score:", round(mean(jlce_scores), 3)))

# Example 2: Bootstrap Confidence Intervals
# ==========================================

# Calculate 95% confidence intervals
bootstrap_results <- bootstrap_jlce_ci(jlce_scores, B = 1000)

cat("\nBootstrap Confidence Interval Results:\n")
cat("Original Mean:", round(bootstrap_results$original_mean, 4), "\n")
cat("95% CI Lower Bound:", round(bootstrap_results$ci_lower, 4), "\n")
cat("95% CI Upper Bound:", round(bootstrap_results$ci_upper, 4), "\n")
cat("Bootstrap Standard Error:", round(bootstrap_results$boot_se, 4), "\n")

# Example 3: Model Comparison Analysis
# ====================================

# Simulate baseline vs optimized model scores
baseline_scores <- jlce_scores + rnorm(length(jlce_scores), -0.05, 0.03)
optimized_scores <- jlce_scores + rnorm(length(jlce_scores), 0.08, 0.04)

# Ensure scores remain in [0,1] range
baseline_scores <- pmax(0, pmin(1, baseline_scores))
optimized_scores <- pmax(0, pmin(1, optimized_scores))

# Perform comparative analysis
comparison_results <- comparative_analysis(baseline_scores, optimized_scores)

cat("\nModel Comparison Results:\n")
cat("Mean Difference:", round(comparison_results$mean_difference, 4), "\n")
cat("Cohen's d (Effect Size):", round(comparison_results$cohen_d, 4), "\n")
cat("Effect Size Interpretation:", comparison_results$effect_size_interpretation, "\n")
cat("Statistically Significant:", comparison_results$significant, "\n")
cat("p-value:", round(comparison_results$p_value, 6), "\n")

# Example 4: Semantic Accuracy Calculation
# =========================================

# Generate sample embeddings (768-dimensional)
n_embeddings <- 10
embedding_dim <- 768

set.seed(123)
expected_embeddings <- matrix(rnorm(n_embeddings * embedding_dim), 
                             nrow = n_embeddings, ncol = embedding_dim)
generated_embeddings <- expected_embeddings + 
                       matrix(rnorm(n_embeddings * embedding_dim, 0, 0.1), 
                             nrow = n_embeddings, ncol = embedding_dim)

# Calculate semantic accuracy
semantic_accuracy <- calculate_semantic_accuracy(expected_embeddings, 
                                                generated_embeddings, 
                                                tau = 0.65)

cat("\nSemantic Accuracy Results:\n")
cat("Mean Semantic Accuracy:", round(mean(semantic_accuracy), 4), "\n")
cat("Min Semantic Accuracy:", round(min(semantic_accuracy), 4), "\n")
cat("Max Semantic Accuracy:", round(max(semantic_accuracy), 4), "\n")

# Example 5: Complete Analysis Pipeline (using synthetic data)
# ============================================================

# Create synthetic evaluation results file
synthetic_results <- list(
  semantic_scores = sample_data$semantic,
  syntactic_scores = sample_data$syntactic,
  pragmatic_scores = sample_data$pragmatic,
  cultural_scores = sample_data$cultural,
  metadata = list(
    model_name = "DeepSeek-R1-Japanese-Adapter",
    evaluation_date = Sys.Date(),
    total_prompts = nrow(sample_data)
  )
)

# Save synthetic data for testing
temp_results_file <- "temp_jlce_results.json"
write_json(synthetic_results, temp_results_file, pretty = TRUE)

# Create synthetic prompts file
synthetic_prompts <- data.frame(
  prompt_id = 1:31,
  domain = rep(c("Technical", "Social Policy", "Emerging Technologies", 
                "Educational", "Infrastructure"), 
               length.out = 31),
  prompt_text = paste("Sample prompt", 1:31),
  expected_output = paste("Expected output", 1:31)
)

temp_prompts_file <- "temp_prompts.jsonl"
write_lines(toJSON(synthetic_prompts, auto_unbox = TRUE), temp_prompts_file)

# Run complete analysis pipeline
cat("\nRunning complete JLCE analysis pipeline...\n")
analysis_results <- run_jlce_analysis(
  results_file = temp_results_file,
  prompts_file = temp_prompts_file,
  output_dir = "example_analysis_output"
)

# Clean up temporary files
file.remove(temp_results_file)
file.remove(temp_prompts_file)

cat("\nExample analysis complete! Check 'example_analysis_output' directory for results.\n")

# Example 6: Visualization Examples
# =================================

if (interactive()) {
  # Create and display dashboard
  dashboard <- create_jlce_dashboard(analysis_results$results)
  
  # Create and display bootstrap CI plot
  ci_plot <- plot_bootstrap_ci(analysis_results$bootstrap)
  
  cat("\nVisualization examples created and displayed!\n")
}

# Example 7: Domain-Specific Analysis
# ===================================

# Analyze performance by domain
if (nrow(analysis_results$results) > 0) {
  
  # Add domain information (synthetic for example)
  analysis_results$results$domain <- rep(c("Technical", "Social Policy", 
                                          "Emerging Technologies", "Educational", 
                                          "Infrastructure"), 
                                        length.out = nrow(analysis_results$results))
  
  # Calculate domain-specific statistics
  domain_stats <- analysis_results$results %>%
    group_by(domain) %>%
    summarise(
      mean_jlce = mean(jlce_percentage),
      median_jlce = median(jlce_percentage),
      sd_jlce = sd(jlce_percentage),
      min_jlce = min(jlce_percentage),
      max_jlce = max(jlce_percentage),
      n_prompts = n()
    )
  
  cat("\nDomain-Specific JLCE Performance:\n")
  print(domain_stats)
}

cat("\n=== All examples completed successfully! ===\n")
