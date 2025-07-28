# AGENTS.md

## Fact_check_pending.md の運用ルール

- CopilotやCodexが仮定で出力した記述・数値を手動/自動でリストアップ
- コードベースの `TODO:` に対応している場合は、ファイル名・行番号を併記
- Codexによる論文更新の際には、必ずここを参照して記述更新の要否を確認
- 実測・確認が必要な数値は、実験後に結果を反映
- 未確認の記述は、`Opinion.md` にて議論・検証

## 統合運用フロー (Updated 2025-07-28)

    ┌─────────────┐         ┌────────────────┐
    │  Codex/Copilot │────→│  TODO コメント追加  │
    └─────────────┘         └────────┬───────┘
                                     ↓
                       tools/scan_todo_codex.py (自動化)
                                     ↓
                    ┌─────────────────────────────┐
                    │  TODO_EXTRACTED.md に一覧出力 │
                    └─────────────────────────────┘
                                     ↓
                          Manual Fact Check or
                          Benchmark Script Update
                                     ↓
                   ┌──────────────────────────────┐
                   │  Fact_check_pending.md を更新  │
                   └──────────────────────────────┘

### 現在の状況 (2025-07-28 22:00)

#### 実装済み TODO 抽出システム
- **tools/scan_todo_codex.py**: 全コードベースのTODO/Copilot指示を自動抽出
- **TODO_EXTRACTED.md**: 抽出されたTODO項目の優先度別整理
- **総計18件のTODO項目**: 論文検証12件、ベンチマーク4件、統計分析2件

#### 緊急実装項目 (Opinion.md R-1~R-8対応)
1. **R-1**: MLA KVキャッシュ効率測定の実装完了必要
2. **R-5/R-6**: LoRA効率性検証のベースライン比較実装
3. **R-3, R-4, R-7, R-8**: paper_validation_suite.py の未実装検証メソッド
4. **統計的検証**: deepseek_r1_statistical_analysis.R の包括的実装

#### 学術的整合性確保
- 全ての論文記載値に対する実証実験実装指示完了
- .copilot-instructions.json による詳細実装ガイド提供
- AGENTS.md遵守事項に基づく文書化プロセス確立


## 環境

### a.開発環境
- OS: Windows 11/Linux/macOS
- プログラミング言語: Python 3.10+(venv使用)
- エディタ: Visual Studio Code

### b.依存関係
- Pythonライブラリ: `torch`, `transformers`, `datasets`, `numpy`, `pandas`
- GPU: 想定せず

### c.実行環境
- Cloud: RunPod
- OS: Linux(Ubuntu 24.04LTS)
- GPU: AMD Instinct MI300X
- CPU : AMD EPYC系
- Backend: ROCm 6.1+
- Python 3.10+

### d.モデル解析
- モデル: DeepSeek R1

目的は、DeepSeek R1モデルの日本語特化チューニングを行うことですので、素の状態のモデルでのSentencePiece-BPEの状況などを解析します。解析にはPythonとRを使用します。分析の結果はAnalyze_DeepSeekR1_Dataディレクトリ以下に保存します。

# 遵守事項
- 開発環境ではvenvを利用すること。
- エラーの原因となるので、絵文字を使わないこと。
- Readmeやドキュメントでも、極力絵文字を使わないこと。
- 論文執筆を意識し、コードやコメントは簡潔に。
- 論文データを扱うため、コードは再現性を重視。
- コードの可読性を重視し、コメントは必要な箇所にのみ、しかし丁寧に記載。
- コードの変更は、必要な場合に限り行うこと。
- コード改変後は、必ずDocs/Descriptions/Description_code_changes_by_agents.MDに変更内容を時系列で記載すること。
- コードの詳細をDocs/Descriptions/Description_codes-[対象コード名].mdに記載すること。変更後の記載忘れに注意。
- コードの詳細をDocs/Descriptions/Description_codes-[対象コード名].mdに記載する際は、コード実装の意図も併せて記載すること。
- コードの詳細Docs/Descriptions/Description_codes-[対象コード名].mdは、コードを引用しつつ、説明文とすること。箇条書きや列挙ばかりしないこと。
- テストコードの説明は、Description_test_codes-[対象コード名].mdに記載すること。
- 説明文を書く際には、人間が読むことを重視し、地の文（printやclassではなく）とすること。
- コード改変や機能更新・追加等での変更があった場合、Docs/Paper_draft以下にある論文の下書きも必要に応じて更新すること。
- 論文下書きは英語と日本語の両方がファイルで分けてあるので、どちらか一方だけを更新するのではなく、両方を更新すること。
- 論文下書きの更新は、コードの変更内容を反映することを目的とし、論文の内容を大きく変更することは避けること。
- 論文下書きの更新で大きく内容を変更すべき場合は、Docs/Paper_draft/Opinion.mdにその内容を記載すること。
- 論文下書き関連ファイルはMarkdown形式で記載すること。
- Opinion.mdは、論文の内容を大きく変更する場合に限らず、気づきを記載する場所としても利用すること。
- Opinion.mdへの記載も、編集日時を付記すること。。
- 作業後、必ずAGENTS.mdの内容に沿ったものかどうかを確認すること。

- 開発環境ではデータセット整形などのために、Python以外にRが使えます。しかし、本番環境ではRは使えません。

- CUDAを一切使用しません。使用できるのはROCMのみです。ただし、CUDA互換レイヤーの使用は許容するため、PyTorchのCUDA APIは使用可能です。
- MetalやDirectMLなど、ROCm以外のGPUアクセラレーションAPIを暫定的に使用する場合は、必ずAGENTS.mdの遵守事項に従い、Docs/Descriptions/Description_code_changes_by_agents.MDに変更内容を記載すること。

# FOR CODEX

# 🔧 INSTRUCTION FOR CODEX (based on AGENTS.md)

## 🧪 Purpose
You are assisting in the development and refinement of a research project to adapt the DeepSeek R1 LLM to Japanese, using LoRA fine-tuning and tokenization analysis. Your job is to:
- Identify and implement unfinished code
- Maintain documentation integrity
- Align code logic with the scientific goals
- Keep reproducibility and readability in mind

## 🛠️ Project Environment

- OS: Windows 11 / Linux (Ubuntu 24.04) / macOS
- Python: 3.10+, with `venv`
- Libraries: `torch`, `transformers`, `datasets`, `numpy`, `pandas`
- GPU: ROCm 6.1+ only (AMD MI300X), **no CUDA**
- Editors: Visual Studio Code
- Runtime: RunPod (Cloud GPU instance)

## 📦 Coding Rules

1. Use `venv` for all development environments.
2. Do NOT use emoji in code or documentation.
3. All comments must be minimal, yet clear and meaningful. Do not over-comment.
4. Do not change the code unless strictly necessary.
5. All code changes must be documented in:  
   `Docs/Descriptions/Description_code_changes_by_agents.md`
6. Any updated or newly written code must be described in:  
   `Docs/Descriptions/Description_codes-[filename].md`  
   Include both code snippets and natural-language explanation.
7. Use descriptive prose. Do NOT rely on bullet-point-only explanations.
8. Unit test explanations go in:  
   `Docs/Descriptions/Description_test_codes-[filename].md`

## 🧾 Documentation Rules

- All changes must be reflected in both English and Japanese versions of the draft under:  
  `Docs/Paper_draft/`
- Do NOT modify the paper’s structure significantly unless justified and recorded in:  
  `Docs/Paper_draft/Opinion.md`
- Use `Opinion.md` also for recording insights, with a timestamp for every entry.

## 🧪 Scientific Constraints

- All code must prioritize reproducibility.
- ROCm is the only allowed GPU backend. CUDA, Metal, DirectML are not allowed.
- Use of CUDA-compatible API within PyTorch is allowed if ROCm-compatible.
- Do not use CUDA directly. If any compatibility layers are used, document them explicitly.

## 🔎 Request

Please:

1. Scan the entire codebase.
2. Identify all `TODO` or `Copilot:` instructions.
3. Follow their description to implement or extend them.
4. Output a unified `TODO.md` and `.copilot-instructions.json` based on these.
5. For any newly implemented or modified logic, update:
   - `Docs/Descriptions/Description_code_changes_by_agents.md`
   - `Docs/Descriptions/Description_codes-[filename].md`
   - `Docs/Paper_draft/[EN|JA].md` (if affects paper contents)
6. Add relevant scientific notes to `Opinion.md` if changes affect methodology.
7. After changes, check conformance with this AGENTS.md.

