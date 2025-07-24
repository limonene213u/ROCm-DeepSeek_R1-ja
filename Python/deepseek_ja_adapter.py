#!/usr/bin/env python3
"""
DeepSeek R1 日本語特化学習 - limo-style版
言語学的特徴を考慮した動的データ生成 + 効率的BPE設計

Author: Akira Ito a.k.a limonene213u
Target: AMD MI300X + ROCm 6.1環境
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import sentencepiece as spm
import fugashi
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import logging
from dataclasses import dataclass
from enum import Enum

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """実行モード"""
    PRODUCTION = "production"    # 本格運用
    DEVELOPMENT = "development"  # 開発・テスト
    TRIAL = "trial"             # 試行・デモ

@dataclass
class JapaneseDataConfig:
    """日本語データセット設定"""
    base_dir: Path = Path("dataset/deepseek-jp")
    train_files: List[str] = None
    validation_files: List[str] = None
    execution_mode: ExecutionMode = ExecutionMode.DEVELOPMENT
    
    def __post_init__(self):
        if self.train_files is None:
            if self.execution_mode == ExecutionMode.PRODUCTION:
                self.train_files = [
                    "wikipedia_ja.jsonl",
                    "cc100_ja.jsonl", 
                    "oscar_ja.jsonl",
                    "aozora_bunko.jsonl",
                    "japanese_news.jsonl",
                    "technical_docs_ja.jsonl"
                ]
            else:
                self.train_files = [
                    "wikipedia_ja.jsonl",
                    "cc100_ja.jsonl", 
                    "conversation_ja.jsonl"
                ]
        
        if self.validation_files is None:
            self.validation_files = ["validation_ja.jsonl"]

class DatasetManager:
    """データセット管理クラス"""
    
    def __init__(self, config: JapaneseDataConfig):
        self.config = config
        self.base_dir = config.base_dir
        
    def ensure_datasets_exist(self) -> bool:
        """データセットの存在確認と必要に応じた生成"""
        
        # 実際のデータセットファイルの確認
        real_files_exist = any(
            (self.base_dir / filename).exists() 
            for filename in self.config.train_files
        )
        
        if real_files_exist:
            logger.info("Real dataset files found. Using existing data.")
            return True
        
        # 本格運用モードでデータセットがない場合はエラー
        if self.config.execution_mode == ExecutionMode.PRODUCTION:
            logger.error("Production mode requires real dataset files.")
            logger.error(f"Please place dataset files in: {self.base_dir}")
            logger.error(f"Expected files: {self.config.train_files}")
            return False
        
        # 開発・試行モードではサンプルデータセットを生成
        logger.info(f"Running in {self.config.execution_mode.value} mode.")
        logger.info("Generating sample datasets for testing...")
        
        self._create_sample_datasets()
        return True
    
    def _create_sample_datasets(self):
        """サンプルデータセット作成"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # テンプレートディレクトリの作成
        template_dir = self.base_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # サンプルテンプレートファイルの作成
        sample_templates = self._get_sample_templates()
        
        # テンプレートファイルを保存
        template_file = template_dir / "text_templates.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(sample_templates, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Template file created: {template_file}")
        
        # サンプル数を実行モードに応じて調整
        if self.config.execution_mode == ExecutionMode.TRIAL:
            num_samples = 50
        else:  # DEVELOPMENT
            num_samples = 200
        
        # テンプレートから動的にサンプルテキストを生成
        sample_texts = self._generate_dynamic_samples(template_file, num_samples)
        
        # 実際のコーパスファイルがあれば、それも読み込む
        real_corpus_samples = self._load_real_corpus_samples()
        if real_corpus_samples:
            sample_texts.extend(real_corpus_samples[:300])
            logger.info(f"Added {len(real_corpus_samples)} samples from real corpus")
        
        # JSONLファイルの作成
        for filename in self.config.train_files:
            filepath = self.base_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                for text in sample_texts:
                    json.dump({"text": text}, f, ensure_ascii=False)
                    f.write('\n')
        
        # ペルソナ設定ファイルも作成
        self._create_persona_config_file()
        
        logger.info(f"Sample datasets created in: {self.base_dir}")
    
    def _get_sample_templates(self) -> Dict:
        """サンプルテンプレート定義"""
        return {
            "conversation_patterns": [
                "こんにちは、{subject}について教えてください。",
                "{topic}について詳しく説明していただけませんか？",
                "今日は{weather}ですね。{activity}はいかがでしょうか？",
                "{question}について、どう思いますか？",
                "ありがとうございます。{feedback}です。"
            ],
            "technical_patterns": [
                "{technology}を使用して{task}を行います。",
                "{language}プログラミングは{adjective}ですね。",
                "{concept}について、わかりやすく説明いたします。",
                "{field}は{description}な分野です。"
            ],
            "variables": {
                "subjects": ["機械学習", "自然言語処理", "プログラミング", "データサイエンス"],
                "topics": ["AI技術", "Python開発", "ディープラーニング", "統計学"],
                "weather": ["良い天気", "雨", "曇り", "晴れ"],
                "activities": ["散歩", "読書", "プログラミング", "勉強"],
                "questions": ["この技術", "この手法", "この概念", "このアプローチ"],
                "feedback": ["とても参考になりました", "勉強になります", "興味深いです"],
                "technologies": ["Python", "PyTorch", "Transformers", "CUDA"],
                "tasks": ["データ解析", "モデル学習", "推論処理", "可視化"],
                "languages": ["Python", "JavaScript", "Go", "Rust"],
                "adjectives": ["楽しい", "興味深い", "有用", "強力"],
                "concepts": ["機械学習", "ニューラルネットワーク", "トランスフォーマー", "BPE"],
                "fields": ["自然言語処理", "コンピュータビジョン", "音声認識", "データマイニング"],
                "descriptions": ["急速に発展している", "重要", "応用範囲の広い", "革新的"]
            }
        }
    
    def _generate_dynamic_samples(self, template_file: Path, num_samples: int) -> List[str]:
        """テンプレートファイルから動的にサンプルを生成"""
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            samples = []
            patterns = templates.get("conversation_patterns", []) + templates.get("technical_patterns", [])
            variables = templates.get("variables", {})
            
            for _ in range(num_samples):
                pattern = random.choice(patterns)
                
                # パターン内の変数を置換
                sample_text = pattern
                for var_name, var_values in variables.items():
                    var_key = var_name[:-1] if var_name.endswith('s') else var_name
                    if f"{{{var_key}}}" in sample_text:
                        replacement = random.choice(var_values)
                        sample_text = sample_text.replace(f"{{{var_key}}}", replacement)
                
                # まだ置換されていない変数があれば、適当な値で置換
                import re
                remaining_vars = re.findall(r'\{(\w+)\}', sample_text)
                for var in remaining_vars:
                    default_values = ["これ", "それ", "例", "もの", "こと"]
                    sample_text = sample_text.replace(f"{{{var}}}", random.choice(default_values))
                
                if sample_text not in samples:  # 重複回避
                    samples.append(sample_text)
            
            logger.info(f"Generated {len(samples)} dynamic samples")
            return samples
            
        except Exception as e:
            logger.warning(f"Template loading error: {e}")
            # フォールバック: 最小限のサンプル
            return [
                "こんにちは、AIアシスタントです。",
                "お手伝いできることがあれば教えてください。",
                "今日も一緒に学びましょう。"
            ] * (num_samples // 3)
    
    def _load_real_corpus_samples(self) -> List[str]:
        """実際のコーパスファイルがあれば読み込む"""
        corpus_samples = []
        
        # 実際のコーパスファイルを探す
        corpus_files = [
            "real_wikipedia_ja.txt",
            "real_cc100_ja.txt", 
            "real_conversations_ja.txt",
            "aozora_bunko_sample.txt"
        ]
        
        for filename in corpus_files:
            filepath = self.base_dir / filename
            if filepath.exists():
                logger.info(f"Loading real corpus: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # 適切な長さのテキストのみ取得
                        for line in lines:
                            line = line.strip()
                            if 10 <= len(line) <= 300:  # 適度な長さ
                                corpus_samples.append(line)
                    
                    logger.info(f"Loaded {len(corpus_samples)} corpus samples")
                except Exception as e:
                    logger.warning(f"Corpus loading error {filepath}: {e}")
        
        return corpus_samples
    
    def _create_persona_config_file(self):
        """ペルソナ設定ファイル作成"""
        persona_config_file = self.base_dir / "persona_config.json"
        
        persona_data = {
            "character_name": "DeepSeek-R1-JP",
            "description": "日本語に特化したAIアシスタント",
            "personality": {
                "価値観": ["親切さ", "正確性", "学習意欲", "創造性"],
                "好物": "新しい知識",
                "口調": "丁寧で親しみやすい"
            },
            "attributes": {
                "専門分野": ["日本語処理", "機械学習", "プログラミング支援", "システム構築"],
                "目的": "皆さんの学習をサポートすること",
                "特徴": ["技術的な質問に詳しい", "日本語の微妙なニュアンスを理解", "実用的なアドバイス"]
            },
            "role": "日本語AI技術アシスタント",
            "background": "プログラミングと機械学習を愛する、日本語特化型のAIアシスタント",
            "conversation_style": {
                "greeting_patterns": [
                    "こんにちは！{character_name}です。",
                    "お疲れさまです。{character_name}がお手伝いします。",
                    "いらっしゃいませ！何かお手伝いできることはありますか？"
                ],
                "response_patterns": [
                    "なるほど、{topic}についてですね。",
                    "とても興味深いご質問です。",
                    "一緒に考えてみましょう。",
                    "お役に立てて嬉しいです。"
                ]
            }
        }
        
        with open(persona_config_file, 'w', encoding='utf-8') as f:
            json.dump(persona_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Persona config file created: {persona_config_file}")

class JapaneseLinguisticProcessor:
    """日本語言語学的処理クラス - limo-style"""
    
    def __init__(self):
        try:
            # fugashiの初期化（NEologd辞書対応）
            import fugashi
            self.tagger = fugashi.Tagger()
            logger.info("fugashi initialized successfully")
            self.available = True
        except ImportError:
            logger.warning("fugashi not installed. Install with: pip install fugashi[unidic-lite]")
            self.available = False
        except Exception as e:
            logger.warning(f"fugashi initialization failed: {e}")
            self.available = False
    
    def morphological_analysis(self, text: str) -> List[Dict]:
        """形態素解析 - fugashi版"""
        if not self.available:
            return [{"surface": text, "pos": "UNKNOWN", "pos_detail": "", "base_form": text}]
        
        morphemes = []
        for word in self.tagger(text):
            # fugashiの特徴量取得
            features = str(word.feature).split(',')
            morphemes.append({
                "surface": word.surface,
                "pos": features[0] if len(features) > 0 else "UNKNOWN",
                "pos_detail": features[1] if len(features) > 1 else "",
                "base_form": features[6] if len(features) > 6 and features[6] != '*' else word.surface,
                "reading": features[7] if len(features) > 7 and features[7] != '*' else word.surface
            })
        
        return morphemes
    
    def generate_linguistic_variants(self, text: str, num_variants: int = 3) -> List[str]:
        """日本語の言語学的特徴を活用したバリアント生成 - limo-style"""
        variants = [text]  # 元テキストも含む
        
        if not self.available:
            return variants
        
        morphemes = self.morphological_analysis(text)
        
        for _ in range(num_variants):
            variant = self._create_sophisticated_variant(morphemes, text)
            if variant and variant != text and variant not in variants:
                variants.append(variant)
        
        return variants
    
    def _create_sophisticated_variant(self, morphemes: List[Dict], original_text: str) -> str:
        """より高度な日本語バリエーション生成"""
        result = []
        skip_next = False
        
        for i, morph in enumerate(morphemes):
            if skip_next:
                skip_next = False
                continue
                
            surface = morph["surface"]
            pos = morph["pos"]
            pos_detail = morph["pos_detail"]
            
            # 1. 動詞の活用形変化（丁寧語⇔普通形）
            if pos == "動詞":
                surface = self._vary_verb_sophisticatedly(surface, morph, morphemes, i)
            
            # 2. 助詞の省略・変更・追加（自然な日本語として）
            elif pos == "助詞":
                surface, skip_next = self._vary_particle_naturally(surface, morphemes, i)
                
            # 3. 形容詞・形容動詞の語尾変化
            elif pos in ["形容詞", "形容動詞"]:
                surface = self._vary_adjective_form(surface, morph)
            
            # 4. 副詞の追加・削除（感情表現豊かに）
            elif pos == "副詞":
                if random.random() < 0.3:  # 30%の確率で削除
                    continue
                surface = self._enhance_adverb(surface)
            
            # 5. 名詞の敬語化・カジュアル化
            elif pos == "名詞":
                surface = self._adjust_noun_politeness(surface, pos_detail)
            
            result.append(surface)
        
        reconstructed = "".join(result)
        
        # 文レベルの調整
        final_text = self._adjust_sentence_level(reconstructed, original_text)
        
        return final_text
    
    def _vary_verb_sophisticatedly(self, surface: str, morph: Dict, morphemes: List[Dict], index: int) -> str:
        """動詞の洗練された活用変化"""
        # だ/である調の変換
        polite_to_casual = {
            "です": ["だ", "である", "なのだ"],
            "でした": ["だった", "であった"],
            "ます": ["る", ""],
            "ました": ["た", "った"],
            "でしょう": ["だろう", "であろう"],
            "ですね": ["だね", "だよね", "ですよね"],
            "ですが": ["だが", "だけど", "ですけど"],
            "ですよ": ["だよ", "だぞ", "なんだよ"]
        }
        
        casual_to_polite = {
            "だ": ["です", "である"],
            "だった": ["でした", "であった"], 
            "だろう": ["でしょう", "であろう"],
            "だね": ["ですね", "ですよね"],
            "だが": ["ですが", "ですけれど"],
            "だよ": ["ですよ", "なのです"]
        }
        
        # より文脈を考慮した変換
        if surface in polite_to_casual:
            variations = polite_to_casual[surface]
            return random.choice(variations)
        elif surface in casual_to_polite:
            variations = casual_to_polite[surface]
            return random.choice(variations)
        
        # 動詞活用の基本パターン
        verb_variations = {
            "思う": ["考える", "感じる"],
            "言う": ["話す", "述べる"],
            "見る": ["眺める", "観る"],
            "聞く": ["尋ねる", "質問する"],
            "する": ["行う", "実行する", "やる"]
        }
        
        return verb_variations.get(surface, surface)
    
    def _vary_particle_naturally(self, surface: str, morphemes: List[Dict], index: int) -> Tuple[str, bool]:
        """助詞の自然な変換"""
        skip_next = False
        
        # 文脈を考慮した助詞変換
        natural_variations = {
            "は": {
                "default": ["が", "も", "って"],
                "topic": ["については", "に関しては"],
                "contrast": ["は", "こそ"]
            },
            "が": {
                "default": ["は", "も"],
                "emphasis": ["こそ", "だって"]
            },
            "を": {
                "default": ["も", "って"],
                "casual": [""],  # 関西弁的省略
            },
            "に": {
                "default": ["へ", "で"],
                "time": ["には", "において"]
            },
            "で": {
                "default": ["において", "にて"],
                "casual": ["だと", "じゃ"]
            }
        }
        
        if surface in natural_variations:
            variations = natural_variations[surface]["default"]
            
            # 省略の可能性（口語的表現）
            if random.random() < 0.15:  # 15%の確率で省略
                return "", skip_next
            
            return random.choice(variations), skip_next
        
        return surface, skip_next
    
    def _vary_adjective_form(self, surface: str, morph: Dict) -> str:
        """形容詞の語尾変化"""
        adjective_variations = {
            # い形容詞
            "美しい": ["きれい", "素敵", "美しい"],
            "大きい": ["でかい", "巨大", "大きな"],
            "小さい": ["ちっちゃい", "小さな", "小柄な"],
            "良い": ["いい", "素晴らしい", "良好"],
            "悪い": ["だめ", "よくない", "不良"],
            
            # な形容詞 
            "簡単": ["楽", "易しい", "シンプル"],
            "複雑": ["難しい", "困難", "厄介"],
            "重要": ["大切", "肝心", "必要"]
        }
        
        return adjective_variations.get(surface, surface)
    
    def _enhance_adverb(self, surface: str) -> str:
        """副詞の表現力向上"""
        adverb_enhancements = {
            "とても": ["非常に", "めちゃくちゃ", "すごく", "かなり"],
            "少し": ["ちょっと", "若干", "やや", "わずかに"],
            "多分": ["おそらく", "たぶん", "きっと", "もしかすると"],
            "すぐに": ["即座に", "直ちに", "さっそく", "早速"]
        }
        
        return random.choice(adverb_enhancements.get(surface, [surface]))
    
    def _adjust_noun_politeness(self, surface: str, pos_detail: str) -> str:
        """名詞の敬語レベル調整"""
        politeness_variations = {
            # 敬語化
            "人": ["方", "お方"],
            "話": ["お話", "お話し"],
            "時間": ["お時間"],
            "会社": ["御社", "貴社"],
            
            # カジュアル化
            "方": ["人"],
            "お話": ["話"],
            "御社": ["会社"],
        }
        
        return politeness_variations.get(surface, surface)
    
    def _adjust_sentence_level(self, text: str, original: str) -> str:
        """文レベルでの調整"""
        # 語尾の統一感調整
        if "です" in text and "だ" in text:
            # 混在を避ける
            if random.random() < 0.5:
                text = text.replace("だ", "です")
            else:
                text = text.replace("です", "だ")
        
        # 感嘆符・疑問符の調整
        emotion_endings = ["。", "！", "♪", "。", "ね。", "よ。"]
        if text.endswith("。") and random.random() < 0.2:
            text = text[:-1] + random.choice(emotion_endings)
        
        return text

class JapaneseBPEStrategy:
    """日本語特化BPE戦略"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.linguistic_processor = JapaneseLinguisticProcessor()
    
    def create_optimized_japanese_bpe(
        self, 
        corpus_files: List[str], 
        vocab_size: int = 32000,
        character_coverage: float = 0.9995
    ) -> str:
        """言語学的特徴を考慮した日本語BPEの作成"""
        logger.info("Creating linguistically-optimized Japanese BPE...")
        
        # 前処理済みコーパスの作成
        preprocessed_corpus = self._preprocess_corpus_for_bpe(corpus_files)
        
        model_prefix = str(self.output_dir / "ja_optimized_bpe")
        
        # SentencePieceの学習パラメータ（日本語最適化）
        smp_params = [
            f'--input={preprocessed_corpus}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={vocab_size}',
            f'--model_type=bpe',
            f'--character_coverage={character_coverage}',
            f'--user_defined_symbols=[PAD],[UNK],[BOS],[EOS],[MASK]',
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3',
            f'--normalization_rule_name=nmt_nfkc_cf',  # 日本語正規化
            f'--remove_extra_whitespaces=false',  # 日本語では重要
            f'--split_by_unicode_script=true',  # 文字種による分割
            f'--split_by_number=true',
            f'--split_by_whitespace=true',
            f'--treat_whitespace_as_suffix=false',
            f'--allow_whitespace_only_pieces=true',
            # 日本語特有の設定
            f'--required_chars=あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー',
            f'--byte_fallback=true'
        ]
        
        smp_command = ' '.join(smp_params)
        spm.SentencePieceTrainer.Train(smp_command)
        
        logger.info(f"Japanese BPE created: {model_prefix}.model")
        return f"{model_prefix}.model"
    
    def _preprocess_corpus_for_bpe(self, corpus_files: List[str]) -> str:
        """BPE学習用のコーパス前処理"""
        preprocessed_file = self.output_dir / "preprocessed_corpus.txt"
        
        with open(preprocessed_file, 'w', encoding='utf-8') as outf:
            for file_path in corpus_files:
                logger.info(f"Processing: {file_path}")
                
                if not Path(file_path).exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        try:
                            if file_path.endswith('.jsonl'):
                                data = json.loads(line.strip())
                                text = data.get('text', '')
                            else:
                                text = line.strip()
                            
                            if text:
                                # 形態素境界を考慮した前処理
                                processed_text = self._preprocess_text_for_bpe(text)
                                outf.write(processed_text + '\n')
                        
                        except Exception as e:
                            logger.warning(f"Skipping malformed line: {e}")
        
        return str(preprocessed_file)
    
    def _preprocess_text_for_bpe(self, text: str) -> str:
        """BPE学習に最適化されたテキスト前処理"""
        # 基本的な正規化
        text = text.strip()
        
        # 形態素解析を活用した境界情報の追加（実験的）
        if self.linguistic_processor.available:
            # 将来的に形態素境界情報を活用した前処理を実装予定
            pass
        
        return text

class DeepSeekJapaneseTrainer:
    """DeepSeek日本語学習クラス - MI300X最適化"""
    
    def __init__(self, config: JapaneseDataConfig, output_dir: str = "./deepseek-ja-output"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.linguistic_processor = JapaneseLinguisticProcessor()
        self.bpe_strategy = JapaneseBPEStrategy(self.output_dir)
        self.dataset_manager = DatasetManager(config)
        
        # MI300X環境チェック
        self._check_mi300x_environment()
    
    def _check_mi300x_environment(self):
        """MI300X環境の確認"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"GPU: {device_name}")
            logger.info(f"GPU Memory: {memory_gb:.1f} GB")
            logger.info(f"ROCm Version: {torch.version.cuda}")
            
            if "MI300X" in device_name:
                logger.info("MI300X detected! Optimizing for high-performance training...")
                self.is_mi300x = True
            else:
                logger.info("Using alternative GPU")
                self.is_mi300x = False
        else:
            logger.warning("No GPU detected")
            self.is_mi300x = False
    
    def _detect_target_modules(self, model) -> List[str]:
        """LoRAターゲットモジュールを検出"""
        logger.info("Detecting LoRA target modules...")
        
        # 既知のモジュール名パターン
        common_patterns = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head", "embed_tokens"
        ]
        
        # モデルの全モジュールを確認
        available_modules = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                module_name = name.split('.')[-1]
                if module_name not in available_modules:
                    available_modules.append(module_name)
        
        logger.info(f"Available modules: {available_modules}")
        
        # 共通パターンとの一致を確認
        target_modules = []
        for pattern in common_patterns:
            if pattern in available_modules:
                target_modules.append(pattern)
        
        # 最低限のモジュール確保
        if not target_modules:
            target_modules = ["q_proj", "v_proj"]  # デフォルト
        
        logger.info(f"Detected target modules: {target_modules}")
        return target_modules
    
    def load_and_prepare_datasets(self) -> Dict[str, Dataset]:
        """データセットの読み込みと準備"""
        logger.info("Loading and preparing Japanese datasets...")
        
        # データセット存在確認と必要に応じた生成
        if not self.dataset_manager.ensure_datasets_exist():
            raise RuntimeError("Failed to ensure dataset availability")
        
        # 学習データの読み込み
        train_texts = []
        for file_name in self.config.train_files:
            file_path = self.config.base_dir / file_name
            if file_path.exists():
                logger.info(f"Loading: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if file_path.suffix == '.jsonl':
                                data = json.loads(line.strip())
                                text = data.get('text', '')
                            else:
                                text = line.strip()
                            
                            if text:
                                train_texts.append(text)
                        
                        except Exception as e:
                            continue
            else:
                logger.warning(f"Dataset file not found: {file_path}")
        
        # 動的データ拡張（日本語言語学的特徴を活用）
        if self.linguistic_processor.available:
            logger.info("Applying linguistic data augmentation...")
            augmented_texts = []
            
            # 実行モードに応じてサンプル数を調整
            if self.config.execution_mode == ExecutionMode.TRIAL:
                sample_limit = min(100, len(train_texts))
            elif self.config.execution_mode == ExecutionMode.DEVELOPMENT:
                sample_limit = min(1000, len(train_texts))
            else:  # PRODUCTION
                sample_limit = len(train_texts)
            
            for text in train_texts[:sample_limit]:
                variants = self.linguistic_processor.generate_linguistic_variants(text, 2)
                augmented_texts.extend(variants)
            
            train_texts.extend(augmented_texts)
            logger.info(f"Data augmented: {len(augmented_texts)} variants added")
        
        # Datasetに変換
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        # バリデーションデータ（簡易版）
        val_texts = train_texts[:min(1000, len(train_texts)//10)]
        val_dataset = Dataset.from_dict({"text": val_texts})
        
        logger.info(f"Datasets prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return {"train": train_dataset, "validation": val_dataset}
    
    def setup_model_and_tokenizer(self, model_name: str = "deepseek-ai/deepseek-r1-distill-qwen-1.5b", continue_from: Optional[str] = None):
        """モデルとトークナイザーのセットアップ - 継続学習対応"""
        logger.info(f"Setting up model: {model_name}")
        
        # 継続学習の場合
        if continue_from:
            logger.info(f"Continue training mode: {continue_from}")
            
            try:
                from peft import PeftModel, PeftConfig
                
                # PeftConfigからベースモデル情報を取得
                peft_config = PeftConfig.from_pretrained(continue_from)
                base_model_name = peft_config.base_model_name_or_path
                
                # ベースモデル読み込み
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    use_cache=False
                )
                
                # 既存のLoRAアダプターを読み込み
                model = PeftModel.from_pretrained(model, continue_from)
                
                # 継続学習のためにアダプターを学習可能にする
                model.train()
                for param in model.parameters():
                    param.requires_grad = False
                
                # LoRAパラメータのみ学習可能にする
                for name, param in model.named_parameters():
                    if any(lora_key in name for lora_key in ["lora_", "adapter", "peft"]):
                        param.requires_grad = True
                
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name,
                    trust_remote_code=True
                )
                
                logger.info("Continue training model loaded successfully")
                return model, tokenizer, True
                
            except Exception as e:
                logger.error(f"Failed to load continue training model: {e}")
                logger.info("Falling back to normal mode...")
        
        # 通常の新規学習
        # 日本語最適化BPEを作成（本格運用時のみ）
        if self.config.execution_mode == ExecutionMode.PRODUCTION:
            corpus_files = [
                str(self.config.base_dir / f) for f in self.config.train_files 
                if (self.config.base_dir / f).exists()
            ]
            
            if corpus_files:
                bpe_model_path = self.bpe_strategy.create_optimized_japanese_bpe(corpus_files)
        
        # DeepSeekモデルの読み込み
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if self.is_mi300x else "eager"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 日本語トークンの追加（必要に応じて）
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 新規学習：LoRA設定が必要
            from peft import LoraConfig, get_peft_model, TaskType
            
            # LoRA対象モジュールの検出
            target_modules = self._detect_target_modules(model)
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # LoRAランク
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=target_modules,
                bias="none"
            )
            
            model = get_peft_model(model, lora_config)
            
            # LoRA設定の確認
            if hasattr(model, 'print_trainable_parameters'):
                model.print_trainable_parameters()
            
            logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer, False
            
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model: {e}")
            logger.info("Using fallback model for testing...")
            
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                torch_dtype=torch.bfloat16 if self.is_mi300x else torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer, False
    
    def train_with_mi300x_optimization(self, model, tokenizer, datasets, epochs=3, learning_rate=2e-4, output_name="deepseek_ja", is_continue_training=False):
        """MI300X最適化された学習"""
        logger.info("Starting MI300X-optimized training...")
        
        # トークナイズ関数
        def tokenize_function(examples):
            max_length = 2048 if self.is_mi300x else 512  # MI300Xなら長いシーケンス
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        
        # データセットのトークナイズ
        tokenized_train = datasets["train"].map(
            tokenize_function, 
            batched=True,
            num_proc=4,  # 並列処理でトークナイズを高速化
            remove_columns=datasets["train"].column_names
        )
        
        tokenized_val = datasets["validation"].map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=datasets["validation"].column_names
        )
        
        # MI300X最適化学習パラメータ
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / output_name),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=8 if self.is_mi300x else 2,
            per_device_eval_batch_size=8 if self.is_mi300x else 2,
            gradient_accumulation_steps=4 if self.is_mi300x else 8,
            gradient_checkpointing=True,
            
            # 学習率・最適化
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=max(100, int(epochs * 50)),
            max_grad_norm=1.0,
            
            # MI300X向け最適化
            fp16=False,
            bf16=True,  # MI300XではBF16推奨
            tf32=True if self.is_mi300x else False,
            dataloader_num_workers=8 if self.is_mi300x else 0,
            dataloader_pin_memory=True,
            
            # 保存・ログ
            save_strategy="steps",
            save_steps=max(200, int(epochs * 100)),
            evaluation_strategy="steps" if len(datasets.get("validation", [])) > 0 else "no",
            eval_steps=max(200, int(epochs * 100)) if len(datasets.get("validation", [])) > 0 else 0,
            logging_strategy="steps",
            logging_steps=50,
            
            # その他
            remove_unused_columns=False,
            report_to=None,
        )
        
        # データコレクター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal Language Modeling
        )
        
        # トレーナー
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )
        
        # 学習実行
        logger.info("Training started...")
        trainer.train()
        
        # モデル保存
        final_model_path = self.output_dir / output_name
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"Training completed! Model saved to: {final_model_path}")
        return str(final_model_path)

def load_persona_data(file_path: Optional[str]) -> Optional[Dict]:
    """ペルソナデータの読み込み"""
    if not file_path or not Path(file_path).exists():
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                content = f.read().strip()
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    f.seek(0)
                    first_line = f.readline().strip()
                    data = json.loads(first_line)
            else:
                data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load persona data: {e}")
        return None

def integrate_persona_training(trainer_instance, persona_data: Dict, base_texts: List[str]) -> List[str]:
    """ペルソナデータを学習データに統合"""
    if not persona_data:
        return base_texts
    
    logger.info("Integrating persona data into training...")
    
    character_name = persona_data.get("character_name", "アシスタント")
    personality = persona_data.get("personality", {})
    attributes = persona_data.get("attributes", {})
    
    # ペルソナ特化の追加学習データ生成
    persona_texts = []
    
    # 自己紹介パターン
    intro_templates = [
        f"こんにちは！{character_name}です。{personality.get('価値観', ['皆さんのお手伝い'])[0]}を大切にしています。",
        f"初めまして、{character_name}と申します。{', '.join(attributes.get('専門分野', ['様々なこと']))}について一緒に学びましょう。",
        f"皆さん、{character_name}です！{personality.get('好物', '学習')}が大好きです。何かご質問はありますか？"
    ]
    
    # 質問応答パターン
    qa_patterns = [
        ("あなたは誰ですか？", f"私は{character_name}です。{persona_data.get('description', 'AIアシスタント')}として活動しています。"),
        ("どんなことが得意ですか？", f"{', '.join(attributes.get('専門分野', ['様々なこと']))}が得意です。お気軽にご質問ください。"),
        ("何を大切にしていますか？", f"{', '.join(personality.get('価値観', ['皆さんのサポート']))}を大切にしています。")
    ]
    
    # ChatML形式で生成
    for template in intro_templates:
        persona_texts.append(f"<|im_start|>assistant\n{template}<|im_end|>")
    
    for user_msg, assistant_msg in qa_patterns:
        conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        persona_texts.append(conversation)
    
    logger.info(f"Generated {len(persona_texts)} persona-related texts")
    
    return base_texts + persona_texts

def main():
    """メイン実行関数 - limo-style + DeepSeek最適化"""
    print("DeepSeek R1 Japanese Adaptation - limo-style Special Edition")
    print("Advanced Linguistic Processing with fugashi + MI300X Optimization")
    print("by limonene213u\n")
    
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek日本語特化学習")
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-r1-distill-qwen-1.5b", help="使用するモデル名")
    parser.add_argument("--continue-from", help="継続学習用のLoRAモデルパス")
    parser.add_argument("--persona-file", help="ペルソナデータファイル")
    parser.add_argument("--epochs", type=int, default=3, help="学習エポック数")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="学習率")
    parser.add_argument("--output-name", help="出力モデル名")
    parser.add_argument("--mode", choices=["production", "development", "trial"], 
                        default="development", help="実行モード")
    parser.add_argument("--auto", action="store_true", help="自動モード")
    args = parser.parse_args()
    
    # 実行モード設定
    execution_mode = ExecutionMode(args.mode)
    
    # 設定
    config = JapaneseDataConfig(execution_mode=execution_mode)
    
    # 学習実行
    trainer = DeepSeekJapaneseTrainer(config)
    
    # ペルソナデータ読み込み
    persona_data = load_persona_data(args.persona_file)
    if persona_data:
        logger.info(f"Persona data loaded: {persona_data.get('character_name', 'Unknown')}")
    
    # データセット準備
    datasets = trainer.load_and_prepare_datasets()
    
    # ペルソナ統合（オプション）
    if persona_data:
        enhanced_texts = integrate_persona_training(trainer, persona_data, datasets["train"]["text"])
        datasets["train"] = Dataset.from_dict({"text": enhanced_texts})
        logger.info(f"Enhanced dataset size: {len(datasets['train'])}")
    
    # モデル・トークナイザー設定
    model, tokenizer, is_continue = trainer.setup_model_and_tokenizer(
        model_name=args.model_name,
        continue_from=args.continue_from
    )
    
    # 学習パラメータ決定
    if args.auto:
        epochs = args.epochs
        learning_rate = args.learning_rate
        output_name = args.output_name or f"deepseek_ja_{'continue' if is_continue else 'new'}"
    else:
        # インタラクティブ設定
        print(f"\nTraining Configuration:")
        epochs = int(input(f"Epochs (default: {args.epochs}): ") or str(args.epochs))
        learning_rate = float(input(f"Learning rate (default: {args.learning_rate}): ") or str(args.learning_rate))
        default_name = f"deepseek_ja_{'continue' if is_continue else 'new'}_{persona_data.get('character_name', 'model') if persona_data else 'model'}"
        output_name = input(f"Output name (default: {default_name}): ") or default_name
    
    # 学習実行
    final_model_path = trainer.train_with_mi300x_optimization(
        model, tokenizer, datasets, 
        epochs=epochs, 
        learning_rate=learning_rate,
        output_name=output_name,
        is_continue_training=is_continue
    )
    
    print(f"\nJapanese-adapted DeepSeek model completed!")
    print(f"Model location: {final_model_path}")
    
    # 簡易テスト
    if not args.auto:
        test_input = input("\nRun model test? (y/N): ")
        if test_input.lower() == 'y':
            simple_test(final_model_path, tokenizer)

def simple_test(model_path: str, tokenizer):
    """簡易的なモデルテスト"""
    try:
        print("Testing model...")
        
        # モデル読み込み
        from peft import PeftModel, AutoModelForCausalLM
        import torch
        
        # ベースモデル取得
        if Path(model_path).exists() and any(Path(model_path).glob("adapter_*")):
            from peft import PeftConfig
            config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        model.eval()
        
        test_prompts = [
            "こんにちは、私は",
            "日本語について",
            "機械学習とは"
        ]
        
        for prompt in test_prompts:
            inputs = tokenizer(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids.cuda(),
                    max_length=inputs.input_ids.shape[1] + 50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            assistant_part = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
            
            print(f"Input: {prompt}")
            print(f"Response: {assistant_part}")
            print()
        
        print("Test completed")
        
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    main()