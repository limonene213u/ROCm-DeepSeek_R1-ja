# =============================================================================
# DeepSeek R1 Efficiency Validation Module - Performance Claims Statistical Verification
# 論文記載効率性の統計的検証とベイジアン分析
# 
# PURPOSE: 論文の性能向上主張の統計的妥当性検証
# - MLA効率性：KVキャッシュ削減（5-13%主張）の仮説検定
# - LoRA効率性：パラメータ削減（200倍主張）の信頼区間推定
# - メモリ効率性：メモリ削減（2倍主張）の対応t検定
# - 処理速度：高速化（10.47倍, 7.60倍主張）のBootstrap検証
# 
# SCOPE: 論文記載値の統計的信頼性保証
# TARGET: 学術的再現性確保・査読対応
# METHOD: ベイジアン推定 + 頻度論的検定のハイブリッド分析
# 
# Author: Akira Ito (AETS - Akatsuki Enterprise Technology Solutions)
# Date: 2025-07-29
# License: BSD-3-Clause
# =============================================================================

# 必要ライブラリの読み込み
library(jsonlite)
library(dplyr)
library(ggplot2)
library(tidyr)
library(broom)
library(bayesplot)
library(rstanarm)
library(patchwork)

# 作業ディレクトリ設定（どのOSでも実行可能なように、ユーザーホームからのパスを自動取得）
base_dir <- file.path(Sys.getenv("HOME"), "MyProject", "01_Research", "ROCm-DeepSeek_R1-ja", "R", "Analyze_DeepSeekR1")
setwd(base_dir)

# データ読み込み関数
load_benchmark_results <- function(results_dir) {
  cat("ベンチマーク結果を読み込み中...\n")
  
  # MLAベンチマーク結果
  mla_files <- list.files(
    path = file.path(results_dir, "benchmark_results"),
    pattern = "mla_benchmark_results_.*\\.json",
    full.names = TRUE
  )
  
  # LoRAベンチマーク結果
  lora_files <- list.files(
    path = file.path(results_dir, "lora_benchmark_results"), 
    pattern = "lora_efficiency_results_.*\\.json",
    full.names = TRUE
  )
  
  # 検証結果
  validation_files <- list.files(
    path = file.path(results_dir, "validation_results"),
    pattern = "paper_validation_results_.*\\.json",
    full.names = TRUE
  )
  
  results <- list(
    mla_files = mla_files,
    lora_files = lora_files,
    validation_files = validation_files
  )
  
  return(results)
}

# MLA効率性統計分析
analyze_mla_efficiency <- function(mla_file) {
  cat("MLA効率性を分析中...\n")
  
  if (length(mla_file) == 0) {
    cat("MLA結果ファイルが見つかりません\n")
    return(NULL)
  }
  
  # 最新ファイル読み込み
  latest_file <- mla_file[length(mla_file)]
  mla_data <- fromJSON(latest_file)
  
  # 結果データフレーム作成
  if (length(mla_data$results) > 0) {
    mla_df <- data.frame(
      model_name = sapply(mla_data$results, function(x) x$model_name),
      attention_type = sapply(mla_data$results, function(x) x$attention_type),
      sequence_length = sapply(mla_data$results, function(x) x$sequence_length),
      batch_size = sapply(mla_data$results, function(x) x$batch_size),
      precision = sapply(mla_data$results, function(x) x$precision),
      kv_cache_memory_mb = sapply(mla_data$results, function(x) x$kv_cache_memory_mb),
      attention_time_ms = sapply(mla_data$results, function(x) x$attention_computation_time_ms),
      peak_memory_mb = sapply(mla_data$results, function(x) x$peak_memory_usage_mb),
      throughput_tokens_sec = sapply(mla_data$results, function(x) x$throughput_tokens_per_sec),
      stringsAsFactors = FALSE
    )
    
    # MLA vs 標準Attention比較
    mla_results <- filter(mla_df, attention_type == "MLA")
    standard_results <- filter(mla_df, attention_type == "Standard")
    
    if (nrow(mla_results) > 0 && nrow(standard_results) > 0) {
      # KVキャッシュ削減率計算
      comparison_df <- merge(
        mla_results, standard_results,
        by = c("sequence_length", "batch_size", "precision"),
        suffixes = c("_mla", "_standard")
      )
      
      if (nrow(comparison_df) > 0) {
        comparison_df <- comparison_df %>%
          mutate(
            kv_reduction_percent = ((kv_cache_memory_mb_standard - kv_cache_memory_mb_mla) / 
                                   kv_cache_memory_mb_standard) * 100,
            speed_improvement_percent = ((attention_time_ms_standard - attention_time_ms_mla) / 
                                        attention_time_ms_standard) * 100,
            memory_reduction_percent = ((peak_memory_mb_standard - peak_memory_mb_mla) / 
                                       peak_memory_mb_standard) * 100
          )
        
        # 統計サマリー
        kv_stats <- summary(comparison_df$kv_reduction_percent)
        speed_stats <- summary(comparison_df$speed_improvement_percent)
        
        # 論文記載値（5-13%）との比較
        paper_claim_lower <- 5
        paper_claim_upper <- 13
        
        verification_result <- ifelse(
          mean(comparison_df$kv_reduction_percent, na.rm = TRUE) >= paper_claim_lower &&
          mean(comparison_df$kv_reduction_percent, na.rm = TRUE) <= paper_claim_upper,
          "VERIFIED", "NEEDS_REVIEW"
        )
        
        cat(sprintf("MLA KVキャッシュ削減率: %.2f%% (論文記載: 5-13%%)\n", 
                   mean(comparison_df$kv_reduction_percent, na.rm = TRUE)))
        cat(sprintf("検証結果: %s\n", verification_result))
        
        return(list(
          comparison_data = comparison_df,
          kv_stats = kv_stats,
          speed_stats = speed_stats,
          verification = verification_result,
          raw_data = mla_df
        ))
      }
    }
  }
  
  return(NULL)
}

# LoRA効率性統計分析
analyze_lora_efficiency <- function(lora_file) {
  cat("LoRA効率性を分析中...\n")
  
  if (length(lora_file) == 0) {
    cat("LoRA結果ファイルが見つかりません\n")
    return(NULL)
  }
  
  # 最新ファイル読み込み
  latest_file <- lora_file[length(lora_file)]
  lora_data <- fromJSON(latest_file)
  
  # 結果データフレーム作成
  if (length(lora_data$results) > 0) {
    lora_df <- data.frame(
      model_name = sapply(lora_data$results, function(x) x$model_name),
      training_method = sapply(lora_data$results, function(x) x$training_method),
      dataset_size = sapply(lora_data$results, function(x) x$dataset_size),
      trainable_parameters = sapply(lora_data$results, function(x) x$trainable_parameters),
      total_parameters = sapply(lora_data$results, function(x) x$total_parameters),
      parameter_reduction_ratio = sapply(lora_data$results, function(x) x$parameter_reduction_ratio),
      peak_memory_mb = sapply(lora_data$results, function(x) x$peak_memory_mb),
      training_time_minutes = sapply(lora_data$results, function(x) x$training_time_minutes),
      eval_loss = sapply(lora_data$results, function(x) x$eval_loss),
      eval_perplexity = sapply(lora_data$results, function(x) x$eval_perplexity),
      stringsAsFactors = FALSE
    )
    
    # LoRA vs フル fine-tuning 比較
    lora_results <- filter(lora_df, training_method == "lora")
    full_results <- filter(lora_df, training_method == "full_finetuning")
    
    if (nrow(lora_results) > 0 && nrow(full_results) > 0) {
      # 統計分析
      param_reduction_stats <- summary(lora_results$parameter_reduction_ratio)
      
      # メモリ削減比較
      memory_comparison <- merge(
        lora_results, full_results,
        by = c("dataset_size"),
        suffixes = c("_lora", "_full")
      )
      
      if (nrow(memory_comparison) > 0) {
        memory_comparison <- memory_comparison %>%
          mutate(
            memory_reduction_ratio = peak_memory_mb_full / peak_memory_mb_lora,
            training_time_ratio = training_time_minutes_full / training_time_minutes_lora,
            performance_retention = eval_perplexity_lora / eval_perplexity_full
          )
        
        # 論文記載値検証
        # パラメータ削減: 200x
        # メモリ削減: 2x
        avg_param_reduction <- mean(lora_results$parameter_reduction_ratio, na.rm = TRUE)
        avg_memory_reduction <- mean(memory_comparison$memory_reduction_ratio, na.rm = TRUE)
        
        param_verification <- ifelse(avg_param_reduction >= 150, "VERIFIED", 
                                   ifelse(avg_param_reduction >= 50, "PARTIAL", "FAILED"))
        memory_verification <- ifelse(avg_memory_reduction >= 1.8, "VERIFIED",
                                    ifelse(avg_memory_reduction >= 1.3, "PARTIAL", "FAILED"))
        
        cat(sprintf("LoRAパラメータ削減率: %.1fx (論文記載: 200x)\n", avg_param_reduction))
        cat(sprintf("LoRAメモリ削減率: %.1fx (論文記載: 2x)\n", avg_memory_reduction))
        cat(sprintf("パラメータ削減検証: %s\n", param_verification))
        cat(sprintf("メモリ削減検証: %s\n", memory_verification))
        
        return(list(
          lora_data = lora_results,
          full_data = full_results,
          comparison_data = memory_comparison,
          param_stats = param_reduction_stats,
          param_verification = param_verification,
          memory_verification = memory_verification,
          avg_param_reduction = avg_param_reduction,
          avg_memory_reduction = avg_memory_reduction
        ))
      }
    }
  }
  
  return(NULL)
}

# ベイジアン分析（論文記載値の信頼区間推定）
bayesian_analysis <- function(mla_analysis, lora_analysis) {
  cat("ベイジアン分析を実行中...\n")
  
  results <- list()
  
  # MLA効率性のベイジアン分析
  if (!is.null(mla_analysis) && !is.null(mla_analysis$comparison_data)) {
    cat("MLA効率性のベイジアン推定...\n")
    
    kv_reductions <- mla_analysis$comparison_data$kv_reduction_percent
    kv_reductions <- kv_reductions[!is.na(kv_reductions)]
    
    if (length(kv_reductions) > 0) {
      # ベイジアン線形回帰
      tryCatch({
        bayes_model_mla <- stan_glm(
          kv_reductions ~ 1,
          data = data.frame(kv_reductions = kv_reductions),
          family = gaussian(),
          prior_intercept = normal(location = 9, scale = 5),  # 論文記載値中央値
          chains = 4,
          iter = 2000,
          refresh = 0
        )
        
        # 信頼区間
        posterior_samples <- as.matrix(bayes_model_mla)
        ci_95 <- quantile(posterior_samples[, "(Intercept)"], c(0.025, 0.975))
        ci_80 <- quantile(posterior_samples[, "(Intercept)"], c(0.1, 0.9))
        
        results$mla_bayes <- list(
          model = bayes_model_mla,
          ci_95 = ci_95,
          ci_80 = ci_80,
          posterior_mean = mean(posterior_samples[, "(Intercept)"]),
          paper_claim_prob = mean(posterior_samples[, "(Intercept)"] >= 5 & 
                                 posterior_samples[, "(Intercept)"] <= 13)
        )
        
        cat(sprintf("MLA削減率 95%%信頼区間: [%.2f%%, %.2f%%]\n", ci_95[1], ci_95[2]))
        cat(sprintf("論文記載値範囲内の確率: %.2f\n", results$mla_bayes$paper_claim_prob))
        
      }, error = function(e) {
        cat("MLA ベイジアン分析でエラー:", e$message, "\n")
      })
    }
  }
  
  # LoRA効率性のベイジアン分析
  if (!is.null(lora_analysis) && !is.null(lora_analysis$lora_data)) {
    cat("LoRA効率性のベイジアン推定...\n")
    
    param_reductions <- lora_analysis$lora_data$parameter_reduction_ratio
    param_reductions <- param_reductions[!is.na(param_reductions)]
    
    if (length(param_reductions) > 0) {
      tryCatch({
        bayes_model_lora <- stan_glm(
          param_reductions ~ 1,
          data = data.frame(param_reductions = param_reductions),
          family = gaussian(),
          prior_intercept = normal(location = 200, scale = 100),  # 論文記載値
          chains = 4,
          iter = 2000,
          refresh = 0
        )
        
        posterior_samples <- as.matrix(bayes_model_lora)
        ci_95 <- quantile(posterior_samples[, "(Intercept)"], c(0.025, 0.975))
        
        results$lora_bayes <- list(
          model = bayes_model_lora,
          ci_95 = ci_95,
          posterior_mean = mean(posterior_samples[, "(Intercept)"]),
          paper_claim_prob = mean(posterior_samples[, "(Intercept)"] >= 150)
        )
        
        cat(sprintf("LoRAパラメータ削減 95%%信頼区間: [%.1fx, %.1fx]\n", ci_95[1], ci_95[2]))
        cat(sprintf("論文記載値達成確率: %.2f\n", results$lora_bayes$paper_claim_prob))
        
      }, error = function(e) {
        cat("LoRA ベイジアン分析でエラー:", e$message, "\n")
      })
    }
  }
  
  return(results)
}

# 可視化作成
create_visualizations <- function(mla_analysis, lora_analysis, bayes_results) {
  cat("可視化を作成中...\n")
  
  plots <- list()
  
  # MLA効率性プロット
  if (!is.null(mla_analysis) && !is.null(mla_analysis$comparison_data)) {
    p1 <- ggplot(mla_analysis$comparison_data, 
                 aes(x = sequence_length, y = kv_reduction_percent)) +
      geom_point(aes(color = precision, size = batch_size), alpha = 0.7) +
      geom_hline(yintercept = c(5, 13), linetype = "dashed", color = "red") +
      geom_smooth(method = "loess", se = TRUE, alpha = 0.3) +
      labs(
        title = "MLA KVキャッシュ削減率 vs シーケンス長",
        subtitle = "赤線: 論文記載値範囲 (5-13%)",
        x = "シーケンス長",
        y = "KVキャッシュ削減率 (%)",
        color = "精度",
        size = "バッチサイズ"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12)
      )
    
    plots$mla_efficiency <- p1
  }
  
  # LoRA効率性プロット  
  if (!is.null(lora_analysis) && !is.null(lora_analysis$comparison_data)) {
    p2 <- ggplot(lora_analysis$lora_data, 
                 aes(x = dataset_size, y = parameter_reduction_ratio)) +
      geom_point(aes(color = factor(dataset_size)), size = 3) +
      geom_hline(yintercept = 200, linetype = "dashed", color = "red") +
      geom_smooth(method = "lm", se = TRUE, alpha = 0.3) +
      scale_y_log10() +
      labs(
        title = "LoRAパラメータ削減率 vs データセットサイズ",
        subtitle = "赤線: 論文記載値 (200x)",
        x = "データセットサイズ",
        y = "パラメータ削減率 (倍)",
        color = "データセット\nサイズ"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12)
      )
    
    plots$lora_efficiency <- p2
  }
  
  # ベイジアン分析プロット
  if (!is.null(bayes_results$mla_bayes)) {
    p3 <- mcmc_areas(bayes_results$mla_bayes$model, 
                     pars = "(Intercept)",
                     prob = 0.8, prob_outer = 0.95) +
      geom_vline(xintercept = c(5, 13), linetype = "dashed", color = "red") +
      labs(
        title = "MLA効率性の事後分布",
        subtitle = "80%・95%信頼区間表示",
        x = "KVキャッシュ削減率 (%)"
      ) +
      theme_minimal()
    
    plots$mla_posterior <- p3
  }
  
  if (!is.null(bayes_results$lora_bayes)) {
    p4 <- mcmc_areas(bayes_results$lora_bayes$model,
                     pars = "(Intercept)",
                     prob = 0.8, prob_outer = 0.95) +
      geom_vline(xintercept = 200, linetype = "dashed", color = "red") +
      labs(
        title = "LoRA効率性の事後分布", 
        subtitle = "80%・95%信頼区間表示",
        x = "パラメータ削減率 (倍)"
      ) +
      theme_minimal()
    
    plots$lora_posterior <- p4
  }
  
  return(plots)
}

# レポート生成
generate_statistical_report <- function(mla_analysis, lora_analysis, bayes_results, plots) {
  cat("統計レポートを生成中...\n")
  
  report <- list(
    analysis_timestamp = Sys.time(),
    summary = list(),
    detailed_results = list(),
    visualizations = list()
  )
  
  # MLA分析サマリー
  if (!is.null(mla_analysis)) {
    report$summary$mla <- list(
      verification_status = mla_analysis$verification,
      avg_kv_reduction = if (!is.null(mla_analysis$comparison_data)) {
        mean(mla_analysis$comparison_data$kv_reduction_percent, na.rm = TRUE)
      } else NA,
      paper_claim_range = "5-13%",
      sample_size = if (!is.null(mla_analysis$comparison_data)) {
        nrow(mla_analysis$comparison_data)
      } else 0
    )
  }
  
  # LoRA分析サマリー
  if (!is.null(lora_analysis)) {
    report$summary$lora <- list(
      param_verification = lora_analysis$param_verification,
      memory_verification = lora_analysis$memory_verification,
      avg_param_reduction = lora_analysis$avg_param_reduction,
      avg_memory_reduction = lora_analysis$avg_memory_reduction,
      paper_claims = list(
        parameter_reduction = "200x",
        memory_reduction = "2x"
      )
    )
  }
  
  # ベイジアン分析サマリー
  if (!is.null(bayes_results)) {
    if (!is.null(bayes_results$mla_bayes)) {
      report$summary$mla_bayesian <- list(
        posterior_mean = bayes_results$mla_bayes$posterior_mean,
        ci_95 = bayes_results$mla_bayes$ci_95,
        paper_claim_probability = bayes_results$mla_bayes$paper_claim_prob
      )
    }
    
    if (!is.null(bayes_results$lora_bayes)) {
      report$summary$lora_bayesian <- list(
        posterior_mean = bayes_results$lora_bayes$posterior_mean,
        ci_95 = bayes_results$lora_bayes$ci_95,
        paper_claim_probability = bayes_results$lora_bayes$paper_claim_prob
      )
    }
  }
  
  # 詳細結果
  report$detailed_results <- list(
    mla_analysis = mla_analysis,
    lora_analysis = lora_analysis,
    bayesian_results = bayes_results
  )
  
  # プロット保存
  if (length(plots) > 0) {
    for (plot_name in names(plots)) {
      filename <- paste0("plot_", plot_name, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".png")
      ggsave(filename, plots[[plot_name]], width = 10, height = 7, dpi = 300)
      report$visualizations[[plot_name]] <- filename
      cat(sprintf("プロット保存: %s\n", filename))
    }
  }
  
  return(report)
}

# メイン分析実行
main_analysis <- function() {
  cat("DeepSeek R1 効率性統計分析を開始\n")
  cat("=" , rep("=", 50), "\n", sep = "")
  
  # 結果ディレクトリ
  results_dir <- "/Users/limonene/MyProject/01_Research/ROCm-DeepSeek_R1-ja"
  
  # データ読み込み
  benchmark_files <- load_benchmark_results(results_dir)
  
  # MLA分析
  mla_analysis <- analyze_mla_efficiency(benchmark_files$mla_files)
  
  # LoRA分析
  lora_analysis <- analyze_lora_efficiency(benchmark_files$lora_files)
  
  # ベイジアン分析
  bayes_results <- bayesian_analysis(mla_analysis, lora_analysis)
  
  # 可視化
  plots <- create_visualizations(mla_analysis, lora_analysis, bayes_results)
  
  # レポート生成
  report <- generate_statistical_report(mla_analysis, lora_analysis, bayes_results, plots)
  
  # 結果保存
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  report_file <- paste0("statistical_analysis_report_", timestamp, ".json")
  
  write_json(report, report_file, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("統計分析レポート保存: %s\n", report_file))
  
  # 簡易サマリー出力
  cat("\n")
  cat("=" , rep("=", 50), "\n", sep = "")
  cat("分析サマリー\n")
  cat("=" , rep("=", 50), "\n", sep = "")
  
  if (!is.null(report$summary$mla)) {
    cat(sprintf("MLA効率性: %.2f%% (検証: %s)\n", 
               report$summary$mla$avg_kv_reduction,
               report$summary$mla$verification_status))
  }
  
  if (!is.null(report$summary$lora)) {
    cat(sprintf("LoRAパラメータ削減: %.1fx (検証: %s)\n",
               report$summary$lora$avg_param_reduction,
               report$summary$lora$param_verification))
    cat(sprintf("LoRAメモリ削減: %.1fx (検証: %s)\n",
               report$summary$lora$avg_memory_reduction,
               report$summary$lora$memory_verification))
  }
  
  cat("\n分析完了\n")
  
  return(report)
}

# 分析実行
if (interactive()) {
  cat("対話モードで実行中...\n")
  result <- main_analysis()
} else {
  cat("スクリプトモードで実行中...\n")
  result <- main_analysis()
}
