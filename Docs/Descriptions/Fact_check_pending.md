# Fact_check_pending.md

最終更新: 2025-07-28 by limonene

## 🧪 実測・確認が必要な数値・主張

### 🔹 MLAの効率性について
- 論文内記述: 「MLAは従来手法の5〜13%にまでKVキャッシュ量を削減」
- 実測: 未実施
- 対応: `attention_eval.py` に `TODO: Benchmark MLA kv_cache_ratio` の記述あり

### 🔹 JLCEがBLEUより日本語に適しているという記述
- 論文内記述: 「JLCEはBLEUより日本語の評価に適する」
- 根拠: 実証論文リンク未記載 / メトリクス比較も未実装
- 対応案: `evaluation.py` にBLEU対比機能追加後に更新予定

---

## 📌 Codex/Copilotが出力した未裏付けの記述

- `Paper_draft/Draft-en.md` の第3章「Tokenizer optimization」で "JLCE achieved 1.4x gain over BLEU" → 根拠となるデータ未添付
- `main_benchmark.py` のコメント "MI300X requires 30% less memory than A100" → スペック上の理論値であり、実測値と異なる可能性あり
