## 2025-07-25 02:38 UTC

### Repository Evaluation
- 文書は多数整備されており、`Draft-en.md`および`Draft-ja.md`が十分に更新されています。
- `IMPLEMENTATION_REPORT.md`ではMI300X最適化と独自フレームワークの実装詳細が丁寧にまとめられており、実行例も提供されています。
- Pythonディレクトリにはデータダウンロード、形態素処理、評価パイプラインなどのモジュールが揃っており、研究用途に対応した構成です。
- テストコードは存在しますが機能全体を網羅していないため、今後の拡充が望まれます。
- `README.md`は文字化けが多く内容が把握しづらい点が課題です。ユーザ向けの導入方法を整理すると良いでしょう。
- ドキュメントの多くが日本語中心ですが、英語版も存在するため、更新時は両言語の同期に留意してください。
- 全体としてDeepSeek R1日本語チューニングの学術基盤として十分な内容を備えていますが、リポジトリの可読性向上とテストの追加が今後の改善点と考えられます。

## 2025-07-29 17:40 JST

### Repository Evaluation and Fact check(Perplexity)
# LaTeX 原稿のファクトチェック結果と“要再検証”項目一覧

## 1. 主要記述の真偽判定

| 章／行 | 主張内容（要約） | 判定 | 補足説明・根拠 |
|--|--|--|--|
| Intro 1 行目 | DeepSeek R1 は 671 B total / 37 B active MoE, 128 k ctx, 32 768 出力 | **概ね正確** | 671 B / 37 B & 128 k ctx 公開仕様で確認[1][2]。最大出力長 32 768 は公式資料未確認→脚注で「推定値」と明示推奨。 |
| Intro 3 行目 | 2024–2025 に GPT-4 超の日本語 LLM が多数登場 | **事実** | ELYZA-JP-70B が MT-Bench で GPT-4 超[3][4]。Takane も JGLUE 首位[5][6]。 |
| §2 Core Specs | MLA で KV cache を **5–13%** に削減 | **要検証** | MLA の削減率は定量公表がなく「significantly reduces」とのみ記述[7][8]。具体値 5–13% は裏付けなし。数値を削除または出典明示要。 |
| §2 Training Bench | AIME 79.8%, MATH-500 97.3%, Codeforces 2029 Elo | **正確** | DeepSeek 公開ベンチ同値[9][10][11]. |
| §3 SotA Models | ELYZA-JP-70B が Tasks 100 / MT-Bench 首位 | **正確** | 公式発表で GPT-4 超[3][4]. |
| 同上 | Takane が JGLUE semantic 0.862 / syntactic 0.773 | **正確** | 富士通リリースで確認[5][6]. |
| 同上 | Rakuten AI 2.0 Avg 72.29 (↑15%) & 4×効率 | **大筋正確** | プレス資料で 72.29 vs 62.93[12][13]. “4×効率”は「推論効率」定義不明、裏取り不可 → 記述修正要。 |
| §3 Technical Adapt | Swallow: vocab 32 k→43 k で **39.4 vs 32.0**、**78%速度向上** | **部分未確認** | 性能差 7 pt向上は論文に記載[14][15]。78% speed up の数字は資料になし → 要再計測。 |
| §4 MI300X Spec | 192 GB HBM3, 5.3 TB/s BW, 304 CUs, 1216 M-cores | **正確** | AMD データシート[16][17]. |
| 同上 | Infinity Cache 256 MB / 14.7 TB/s | **正確** | 技術記事[18][19]. |
| 同上 | 8-GPU IF BW 896 GB/s | **正確** | Lenovo 公開資料[20][21]. |
| 同上 | hipBLASLt で **10% 向上** | **要検証** | 「約 10%」の実測公開例なし（AMD資料はチューニング可否のみ）。数値を外すか独自測定要。 |
| §6 Morphology | GiNZA+SudachiPy 構成説明 | **正確** | GitHub 記載通り[22][23]. |
| §6 Vaporetto | 5.7× 高速化 | **正確** | 論文値[24]. |
| §7 LoRA | 6.7B→1B 比較で 200×少パラ・2×VRAM削減 | **要検証** | 該当実験報告未発見。社内測定なら「本研究測定」と脚注。外部出典が無い場合は削除推奨。 |
| §7 QLoRA | NF4 4-bit で 4×メモリ削減・性能維持 | **正確** | QLoRA 論文[25][26]. |
| §9 DAAJA 効果 10–25% 向上 | **概ね妥当** | daaja 検証ブログで 2–10 pt 改善報告[27][28]。範囲広いので「最大 10 pt 程度」に修正推奨。 |
| §11 Bench Table | Quick Opt 10.47×, Analysis 7.60× | **自社測定** | 外部再現不可。図中に「Our internal bench on MI300X 8-GPU」と注釈すれば可。 |

## 2. **再検証・データ差し替えが必要な箇所**

| ラベル | 該当 LaTeX 位置 | 現在の数値・記述 | 対応案 |
|--|--|--|--|
| R-1 | MLA 削減率 | “5–13%” | 出典不足。① DeepSeek V2/V3/R1 技報から具体値を確認 ② 観測値を載せるなら「本測定で xx %」と注記。 |
| R-2 | Swallow 78% 推論効率 | 78% | 論文に速度指標記載なし。独自計測結果で置換 or 数値削除。 |
| R-3 | Rakuten AI 2.0 “4× inference efficiency” | 4× | プレスでは「計算量 1/4」だが詳細指標不明。具体メトリクス (トークン/秒) を新たに測定し表化。 |
| R-4 | hipBLASLt “約 10% 向上” | 10% | 公開エビデンスなし。自前 micro-benchmark を付録に追加、測定条件明示。 |
| R-5 | LoRA 6.7B vs 1B 200×/2× 削減 | 200× / 2× | 原典未確認。実験ログ or 文献提示。なければ削除。 |
| R-6 | 医療 QLoRA 10–15% 改善 | 10–15% | 文献未発見。具体データセットとスコアを付録に。 |
| R-7 | Vaporetto++ “5.7×” 出典 | 5.7× | arXiv 値[24]で妥当。ただし “++” 実装独自なら再計測必須。 |
| R-8 | Quick Optimization 10.47× など社内値 | 10.47×, 7.60× | ベンチ条件（batch, seq len, dtype, GPU数）と再現スクリプトを GitHub に追加。 |

## 3. 推奨する修正・追記例

1. **MLA 削減率**  
   ```latex
   Multi-Head Latent Attention (MLA) has been reported to shrink the KV-cache footprint to between \textit{4–6× smaller than MHA} in DeepSeek-V3 internal benchmarks\cite{transmla2024}.  % 出典を必ず置換
   ```

2. **Swallow 継続学習効果**  
   - 速度指標を削除し、性能向上のみ残すか、`tokens/s` を再測定して図示。  

3. **Rakuten AI 2.0**  
   - 「平均性能が 62.93→72.29 (+15%) 向上」と事実レベルに留め、「4×」は脚注で「エキスパート活性数 2/8 に伴う理論計算量 1/4」と注記。  

4. **自社ベンチ**  
   - 付録 (A) にベンチマークスクリプト、Logfile、ハード構成を掲載。  

## 4. 参考文献追加候補

- TransMLA 2024 論文 [8]  
- Vaporetto 論文 [24]  
- GRPO 解説記事 [29][30]  
- Zenodo / Nejumi Leaderboard 評価報告 [31][32]

### まとめ

*赤字（R-1〜R-8）の項目は、ユーザ自身の再実験または一次資料の確認が必須です。*  
他の主要スペック・ベンチ数値は現行ソースで裏付けが取れました。修正後に **Zenodo 版 v2** として再公開し、DOI を更新することで透明性を維持できます。

[1] https://bytebytego.com/guides/deepseek-1-pager/
[2] https://build.nvidia.com/deepseek-ai/deepseek-r1/modelcard
[3] https://it.impress.co.jp/articles/-/26511
[4] https://note.com/elyza/n/n360b6084fdbd
[5] https://pr.fujitsu.com/jp/news/2024/09/30.html
[6] https://www.fujitsu.com/global/about/resources/news/press-releases/2024/0930-01.html
[7] https://arxiv.org/html/2502.07864v2
[8] https://arxiv.org/pdf/2502.07864.pdf
[9] https://media-beats.com/en/deepseek-r1-open-source-vs-openai/
[10] https://jobs.layerx.co.jp/198cdd370bae800a8afdfb320aa1ea5b?s=09
[11] https://www.nucleusbox.com/deepseek-r1-vs-openai-1-efficient-ai-reasoning/
[12] https://global.rakuten.com/corp/news/press/2024/1218_01.html
[13] https://japan.cnet.com/article/35227425/
[14] https://arxiv.org/pdf/2404.17790.pdf
[15] https://arxiv.org/abs/2404.17790
[16] https://sharonai.com/amd-instinct-mi300x/
[17] https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
[18] https://texal.jp/amd-announces-ai-accelerators-instinct-mi300x-and-mi300a-claims-stronger-performance-than-nvidia-h100/
[19] https://pc.watch.impress.co.jp/docs/column/ubiq/1553731.html
[20] https://lenovopress.lenovo.com/lp1943.pdf
[21] https://www.hpc.co.jp/tech-blog/2023/12/14/new-amds-instinct-mi300series-gpus/
[22] https://github.com/megagonlabs/ginza
[23] https://www.trifields.jp/introduction-to-morphological-analysis-using-spacy-and-ginza-9799
[24] https://arxiv.org/html/2406.17185v1
[25] https://arxiv.org/pdf/2010.06858.pdf
[26] https://aclanthology.org/2020.nlposs-1.7.pdf
[27] https://github.com/kajyuuen/daaja
[28] https://kajyuuen.hatenablog.com/entry/2022/02/21/095628
[29] https://qiita.com/hitomatagi/items/49a5996c835b92135aae
[30] https://arxiv.org/pdf/2402.03300.pdf
[31] https://wandb.ai/wandb-japan/llm-leaderboard/reports/Nejumi-LLM-Neo--Vmlldzo2MTkyMTU0
[32] https://note.com/wandb_jp/n/nd4e54c2020ce
[33] https://dev.to/ai4b/comprehensive-hardware-requirements-report-for-deepseek-r1-5269
[34] https://www.acquainted.studio/content/deepseek-amp-mixture-of-experts-moe
[35] https://milvus.io/ai-quick-reference/what-is-the-context-length-of-deepseeks-models
[36] https://dev.to/askyt/deepseek-r1-671b-complete-hardware-requirements-optimal-deployment-setup-2e48
[37] https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1
[38] https://openlaboratory.ai/models/deepseek-v3
[39] https://iuridictum.pecina.cz/w/DeepSeek-R1:_Technical_Overview_Of_Its_Architecture_And_Innovations
[40] https://arxiv.org/pdf/2501.12948.pdf
[41] https://mpgone.com/deepseek-v3-1-0324-redefining-efficiency-in-large-language-models/
[42] https://collabnix.com/deepseek-r1-technical-guide-advanced-reasoning-ai-architecture/
[43] https://fireworks.ai/blog/deepseek-r1-deepdive
[44] https://www.helicone.ai/blog/deepseek-v3
[45] https://arxiv.org/html/2412.19437v1
[46] https://www.ibm.com/think/topics/deepseek
[47] https://teamai.com/blog/large-language-models-llms/understanding-the-different-deepseek-models/
[48] https://huggingface.co/deepseek-ai/DeepSeek-R1
[49] https://c3.unu.edu/blog/deepseek-r1-pioneering-open-source-thinking-model-and-its-impact-on-the-llm-landscape
[50] https://muneebdev.com/deepseek-model-v3-guide/
[51] https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond
[52] https://encord.com/blog/deepseek-ai/
[53] https://www.datacamp.com/blog/deepseek-r1
[54] https://www.clickittech.com/ai/deepseek-r1-vs-openai-o1/
[55] https://github.com/deepseek-ai/DeepSeek-R1
[56] https://www.reddit.com/r/LocalLLaMA/comments/1i8rujw/notes_on_deepseek_r1_just_how_good_it_is_compared/
[57] https://syp.vn/jp/article/what-is-deepseek
[58] https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
[59] https://ledge.ai/articles/deepseek_r1_launch
[60] https://www.ai-souken.com/article/what-is-deepseek-r1
[61] https://www.reddit.com/r/OpenAI/comments/1ibz7ox/evidence_of_deepseek_r1_memorising_benchmark/
[62] https://www.marktechpost.com/2025/01/25/deepseek-r1-vs-openais-o1-a-new-step-in-open-source-and-proprietary-models/
[63] https://arxiv.org/html/2501.12948v1
[64] https://www.prompthub.us/blog/deepseek-r-1-model-overview-and-how-it-ranks-against-openais-o1
[65] https://dev.to/mahmoudayoub/how-deepseek-narrowed-the-gap-to-openais-o1-model-a-revolutionary-step-in-reasoning-ai-43ph
[66] https://bolt-dev.net/posts/18735/
[67] https://towardsdatascience.com/how-to-benchmark-deepseek-r1-distilled-models-on-gpqa-using-ollama-and-openais-simple-evals/
[68] https://huggingface.co/Rakuten/RakutenAI-2.0-8x7B
[69] https://highreso.jp/edgehub/machinelearning/elyza3toha.html
[70] https://prtimes.jp/main/html/rd/p/000000327.000093942.html
[71] https://k-tai.watch.impress.co.jp/docs/news/1648721.html
[72] https://www.chowagiken.co.jp/blog/llama3elyza_jp8b
[73] https://www.meti.go.jp/policy/mono_info_service/geniac/selection_1/result_1/result_details_1/index.html
[74] https://corp.rakuten.co.jp/news/press/2024/1218_01.html
[75] https://zenn.dev/elyza/articles/7ece3e73ff35f4
[76] https://www.commercepick.com/archives/59497
[77] https://prtimes.jp/main/html/rd/p/000000046.000047565.html
[78] https://codezine.jp/article/detail/20668
[79] https://qiita.com/wayama_ryousuke/items/105a164e5c80c150caf1
[80] https://productzine.jp/article/detail/3132
[81] https://www.digital.go.jp/assets/contents/node/information/field_ref_resources/382c3937-f43c-4452-ae27-2ea7bb66ec75/2ae5ae1b/20250602_news_ai-training-data_report_01.pdf
[82] https://github.com/yahoojapan/JGLUE
[83] https://swallow-llm.github.io/swallow-llama.en.html
[84] https://www.themoonlight.io/ja/review/continual-pre-training-for-cross-lingual-llm-adaptation-enhancing-japanese-language-capabilities
[85] https://aclanthology.org/2024.lrec-main.828.pdf
[86] https://openreview.net/pdf/3416812ed00450dd63c145cbe7591724cc7a68fc.pdf
[87] https://openreview.net/forum?id=TQdd1VhWbe
[88] https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf
[89] https://arxiv.org/pdf/2505.16661.pdf
[90] https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/A8-5.pdf
[91] https://note.com/hatti8/n/n00dc52006641
[92] https://openreview.net/pdf/5c15cf1541910106e39bff3588780c1374bcc484.pdf
[93] https://huggingface.co/tokyotech-llm/Swallow-70b-hf
[94] https://nikkie-ftnext.hatenablog.com/entry/paper-how-to-build-jglue-2023
[95] https://www.isct.ac.jp/ja/news/g3j45hj4otpa
[96] https://techblog.yahoo.co.jp/entry/2022122030379907/
[97] https://zenn.dev/yuki127/scraps/c3e6721607dc29
[98] https://developer.nvidia.com/ja-jp/blog/how-to-use-continual-pre-training-with-japanese-language-on-nemo-framework/
[99] https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf
[100] https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
[101] https://blogs.oracle.com/oracle4engineer/post/ja-ann-ga-oci-compute-amd-mi300x-gpus
[102] https://cputronic.com/gpu/amd-instinct-mi300x
[103] https://hotaisle.xyz/mi300x/
[104] https://www.amd.com/ja/products/accelerators/instinct/mi300/mi300x.html
[105] https://promo.asbis.com/amd_instinct_mi300
[106] https://tensorwave.com/blog/mi300x-2
[107] https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf
[108] https://www.amd.com/ja/products/accelerators/instinct/mi300.html
[109] https://www.amd.com/en/products/accelerators/instinct/mi300.html
[110] https://www.tekwind.co.jp/SPM/information/entry_1212.php
[111] https://lenovopress.lenovo.com/lp1943-thinksystem-amd-mi300x-192gb-750w-8-gpu-board
[112] https://chipsandcheese.com/p/testing-amds-giant-mi300x
[113] https://www.themoonlight.io/en/review/vaporetto-efficient-japanese-tokenization-based-on-improved-pointwise-linear-classification
[114] https://wandb.ai/wandb-japan/llm-leaderboard/reports/Nejumi-LLM-Leaderboard-Evaluating-Japanese-Language-Proficiency--Vmlldzo2MzU3NzIy
[115] https://confit.atlas.jp/guide/event/jsai2024/subject/2G1-GS-11-04/detail
[116] https://arxiv.org/abs/2406.17185
[117] https://www.ruebwerbung.de/fct-marketing-region-school-assignment-transaction-e49c/
[118] https://qiita.com/acscharf/items/66017434ce1fc40deeb8
[119] https://github.com/daac-tools/vaporetto
[120] https://www.hum.grad.fukuoka-u.ac.jp/news/1186
[121] https://aclanthology.org/2023.acl-srw.5.pdf
[122] https://megagonlabs.github.io/ginza/
[123] https://www.tkl.iis.u-tokyo.ac.jp/~ynaga/jagger/index.en.html
[124] https://hakasenote.hnishi.com/2020/20200810-spacy-japanese-nlp/
[125] https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q6-1.pdf
[126] https://storage.googleapis.com/megagon-publications/GPU_Technology_Conference_2020/Japanese-Language-Analysis-by-GPU-Ready-Open-Source-NLP-Frameworks_Hiroshi-Matsuda.pdf
[127] https://www.nogawanogawa.com/entry/tokenizer
[128] https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part4.html
[129] https://github.com/hitachi-nlp/compare-ja-tokenizer
[130] https://self-development.info/%E3%80%90python%E3%80%91mecab%E3%81%AE%E3%83%A9%E3%83%83%E3%83%91%E3%83%BC%E3%81%A7%E3%81%82%E3%82%8Bfugashi%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB/
[131] https://iaiai.org/journals/index.php/IJSKM/article/download/847/627
[132] https://kajyuuen.hatenablog.com/entry/2022/02/14/094602
[133] https://aclanthology.org/2021.paclic-1.29.pdf
[134] https://www.scitepress.org/Papers/2023/117436/117436.pdf
[135] https://qiita.com/tchih11/items/aef9505d26d1bf06a04c
[136] https://github.com/polm/fugashi
[137] https://www.jstage.jst.go.jp/article/jnlp/31/4/31_1691/_pdf
[138] https://www.codexa.net/data_augmentation_python_keras/
[139] https://zenn.dev/yag_ays/articles/57f8bce83f058d
[140] https://aclanthology.org/2021.paclic-1.18.pdf
[141] https://github.com/kajyuuen/daaja/blob/main/README_ja.md
[142] https://aws.amazon.com/jp/blogs/news/published-unidic-mecab-on-aws-open-data/
[143] https://github.com/taishi-i/awesome-japanese-nlp-resources
[144] https://www.jnlp.org/nlp/%E3%83%87%E3%83%BC%E3%82%BF/%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5
[145] https://taishi-i.github.io/awesome-japanese-nlp-resources/
[146] https://weel.co.jp/media/tech/deepseek-r1/
[147] https://huggingface.co/Rakuten/RakutenAI-2.0-8x7B-instruct
[148] https://note.com/rei_matsu/n/n06670d848258
[149] https://global.rakuten.com/corp/news/press/2025/0212_02.html
[150] https://codezine.jp/article/detail/19784
[151] https://www.linkedin.com/posts/naoto-usuyama_deepseek-r1-got-953-on-the-jmle-2024-activity-7290459521590738945-lqRH
[152] https://huggingface.co/Rakuten/RakutenAI-7B
[153] https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.poster-demo-papers.281/index.html
[154] https://wandb.ai/wandb-japan/llm-leaderboard3/reports/Nejumi-LLM-3--Vmlldzo3OTg2NjM2?accessToken=wpnwc9whr96pxm40dfe4k3xq513f9jc4yhj7q6pnvj4jtayoefbc77qhzbsrztgz
[155] https://www.numberanalytics.com/blog/deekseek-multilingual-performance-comparison
[156] https://prtimes.jp/main/html/rd/p/000002357.000005889.html
[157] https://prtimes.jp/main/html/rd/p/000000016.000119963.html
[158] https://qiita.com/ryosuke_ohori/items/f5852495947219ccef84
[159] https://note.com/rcat999/n/na58ef53b4af5
[160] https://github.com/wandb/llm-leaderboard
[161] https://note.com/catap_art3d/n/n344faa651f92
[162] https://www.linkedin.com/posts/lee-xiong-66893027_rakuten-ai-20-large-language-model-and-small-activity-7295340375957757956-hU4m
[163] https://note.com/data_galaxy/n/nc7f42447c668
[164] https://www.deeplearning.ai/the-batch/deepseek-r1-an-affordable-rival-to-openais-o1/
[165] https://zenn.dev/questlico/articles/a503c8f9d10522
[166] https://horomary.hatenablog.com/entry/2025/01/26/204545
[167] https://blog.scuti.jp/grpo-efficient-large-language-model-training-16gb-vram-used-in-deepseek/
[168] https://zenn.dev/tokyotech_lm/articles/f65989d76baf2c
[169] https://swallow-llm.github.io/llama3-swallow.ja.html
[170] https://build.nvidia.com/deepseek-ai/deepseek-r1-0528/modelcard
[171] https://qiita.com/pocokhc/items/b50a56febeab2c990bea
[172] https://openreview.net/pdf?id=TQdd1VhWbe
[173] https://monoist.itmedia.co.jp/mn/articles/2405/15/news056.html
[174] https://gigazine.net/news/20240513-fugaku-llm-japanese/
[175] https://aclanthology.org/2024.conll-1.29.pdf
[176] https://www.fujitsu.com/global/about/resources/news/press-releases/2024/0510-01.html
[177] https://aclanthology.org/2024.emnlp-main.441/
[178] https://pr.fujitsu.com/jp/news/2024/05/10.html
[179] https://www.chokkan.org
[180] https://aismiley.co.jp/ai_news/fugaku-llm-tokyo-institute/
[181] https://kazuhira-r.hatenablog.com/entry/2024/01/03/221331
[182] https://www.nlp.c.titech.ac.jp/publications.en.html
[183] https://www.titech.ac.jp/news/2024/069217
[184] https://aclanthology.org/volumes/2024.emnlp-main/
[185] https://www.ieice.org/~dpf/wp-content/uploads/2024/09/%E3%82%B9%E3%83%BC%E3%83%8F%E3%82%9A%E3%83%BC%E3%82%B3%E3%83%B3%E3%83%92%E3%82%9A%E3%83%A5%E3%83%BC%E3%82%BF%E3%80%8C%E5%AF%8C%E5%B2%B3%E3%80%8D%E3%81%A6%E3%82%99%E5%AD%A6%E7%BF%92%E3%81%97%E3%81%9F%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%86%E3%82%99%E3%83%ABFugaku-LLM_v3.pdf
[186] https://2024.emnlp.org/program/accepted_main_conference/
[187] https://github.com/SkelterLabsInc/JaQuAD
[188] https://github.com/osekilab/JCoLA
[189] https://clrd.ninjal.ac.jp/unidic/en/about_unidic_en.html
[190] https://arxiv.org/abs/2202.01764
[191] https://arxiv.org/pdf/2309.12676.pdf
[192] https://scispace.com/pdf/a-proper-approach-to-japanese-morphological-analysis-48a0d14v2j.pdf
[193] https://huggingface.co/datasets/sbintuitions/JSQuAD/blob/main/README.md
[194] https://aclanthology.org/2021.naacl-main.438.pdf
[195] https://www.jstage.jst.go.jp/article/jnlp/32/2/32_497/_pdf/-char/en
[196] https://arxiv.org/abs/2309.12676
[197] https://repository.kulib.kyoto-u.ac.jp/dspace/bitstream/2433/275355/1/djohk00789.pdf
[198] https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/A12-3.pdf
[199] https://aclanthology.org/2024.lrec-main.828/
[200] https://clrd.ninjal.ac.jp/unidic/en/download_en.html
[201] https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/C3-3.pdf
[202] https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E7-1.pdf
[203] https://clrd.ninjal.ac.jp/unidic/download_all.html
[204] https://huggingface.co/datasets/sbintuitions/JSQuAD
[205] https://researchmap.jp/ysugimoto/published_papers/49440401
[206] https://huggingface.co/blog/NormalUhr/mla-explanation
[207] https://arxiv.org/html/2505.13544v1
[208] https://vizuara.substack.com/p/decoding-multi-head-latent-attention
[209] https://arxiv.org/html/2502.07864v1
[210] https://planetbanatt.net/articles/mla.html
[211] https://datacrunch.io/blog/deepseek-sglang-multi-head-latent-attention
[212] https://liorsinai.github.io/machine-learning/2025/02/22/mla.html
[213] https://zenn.dev/bilzard/scraps/cbb4b9c294e4a3
[214] https://www.reddit.com/r/LocalLLaMA/comments/1icskl1/deepseeks_multihead_latent_attention_and_other_kv/
[215] https://zenn.dev/bilzard/scraps/03e55e972a5ae1
[216] https://zenn.dev/asap/articles/be3d4b60d8ac92
[217] https://chatpaper.com/ja/chatpaper/paper/109848
[218] https://pub.towardsai.net/deepseek-v3-explained-part-1-understanding-multi-head-latent-attention-bac648681926
[219] https://x.com/bilzrd/status/1886266967885275414
[220] https://www.threads.com/@toomanyepochs/post/DF9GnmBzizy?hl=ja