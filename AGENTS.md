# AGENTS.md

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

- Appendixには、AI研究者ではない人にもわかるような説明を加えること。目的別・言語（日英）別にファイルを用意しているので、上記に準じるポリシーで作成してください。ただし、小学生にもわかるような懇切丁寧な説明を心がけてください。非AI研究者にも伝わるよう、懇切丁寧な説名をお願いします。
- Appendixを加筆する場合は、必ず元コードやDraft内容に準拠して記載します。元コードやDraftを破壊してはいけません。

- 開発環境ではデータセット整形などのために、Python以外にRが使えます。しかし、本番環境ではRは使えません。

- CUDAを一切使用しません。使用できるのはROCMのみです。ただし、CUDA互換レイヤーの使用は許容するため、PyTorchのCUDA APIは使用可能です。
- MetalやDirectMLなど、ROCm以外のGPUアクセラレーションAPIを暫定的に使用する場合は、必ずAGENTS.mdの遵守事項に従い、Docs/Descriptions/Description_code_changes_by_agents.MDに変更内容を記載すること。