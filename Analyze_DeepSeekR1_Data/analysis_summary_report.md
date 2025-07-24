# DeepSeek R1 Models: Japanese Tokenizer Analysis Report

Analysis Date: 2025-07-25 02:34:03

## Executive Summary

This report analyzes the Japanese language support in DeepSeek R1 Distill models' tokenizers.

## Model Comparison Overview

```
                   model_name  vocab_size  japanese_token_count  japanese_ratio  hiragana_count  katakana_count  kanji_count  mixed_count  byte_token_count  avg_compression_ratio  compression_ratio_std  avg_tokens_per_sentence  avg_chars_per_token  common_word_coverage_rate  mean_token_length  median_token_length
deepseek-r1-distill-qwen-1.5b      151665                     0             0.0               0               0            0            0                 0               0.741071               0.146268                   10.375             1.349398                        0.0           6.436097                    0
```

## Key Findings

- **Best Japanese Support**: deepseek-r1-distill-qwen-1.5b (0.000 Japanese token ratio)
- **Most Efficient Compression**: deepseek-r1-distill-qwen-1.5b (0.741 tokens/char)
- **Best Common Word Coverage**: deepseek-r1-distill-qwen-1.5b (0.000 coverage rate)

## Recommendations

Based on the analysis results:

1. For Japanese text processing, consider the model with highest Japanese token ratio
2. For efficiency, consider the model with lowest compression ratio
3. For general Japanese usage, consider the model with highest common word coverage
