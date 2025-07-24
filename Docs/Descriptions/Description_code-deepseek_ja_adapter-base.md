# deepseek_ja_adapter.py コード詳細説明

## 概要

deepseek_ja_adapter.pyは、DeepSeek R1モデルの日本語特化チューニングを実現するメインスクリプトである。言語学的特徴を考慮した動的データ生成機能、効率的なBPE（Byte Pair Encoding）設計、AMD MI300X + ROCm 6.1環境での最適化された学習パイプラインを統合的に提供する。このスクリプトは、日本語の複雑な言語現象に対応した高品質な言語モデル構築を目的として設計されている。

## 実装意図

本スクリプトの主要な設計思想は、日本語特有の言語学的特徴を深く理解し、それを言語モデルの学習プロセスに効果的に反映させることである。従来の多言語モデルでは十分に捉えきれない日本語の敬語システム、助詞の複雑な使い分け、語順の柔軟性、文脈依存的な語彙選択などを、形態素解析技術と動的データ拡張により学習データに組み込んでいる。また、ROCm環境での効率的な学習を実現するため、メモリ最適化とGPUリソースの最大活用を図った実装となっている。

## クラス設計とアーキテクチャ

### ExecutionMode 列挙型

```python
class ExecutionMode(Enum):
    """実行モード"""
    PRODUCTION = "production"    # 本格運用
    DEVELOPMENT = "development"  # 開発・テスト
    TRIAL = "trial"             # 試行・デモ
```

実行環境に応じた動作制御を行う列挙型である。本格運用時の厳格なデータセット要件から、開発時の柔軟なテスト環境まで、用途に応じた適切な設定を自動選択する仕組みを提供している。

### JapaneseDataConfig データクラス

```python
@dataclass
class JapaneseDataConfig:
    """日本語データセット設定"""
    base_dir: Path = Path("dataset/deepseek-jp")
    train_files: List[str] = None
    validation_files: List[str] = None
    execution_mode: ExecutionMode = ExecutionMode.DEVELOPMENT
    require_all_files: bool = True
```

日本語学習データセットの構成を管理するデータクラスである。実行モードに応じて必要なデータセットファイルを動的に決定し、本番環境では包括的なデータセット（Wikipedia、CC-100、OSCAR、青空文庫、ニュース、技術文書）を、開発環境では基本的なセット（Wikipedia、CC-100、対話データ）を要求する。これにより、開発効率と本番品質の両立を実現している。

## データセット管理システム

### DatasetManager クラス

```python
class DatasetManager:
    """データセット管理クラス"""
    
    def __init__(self, config: JapaneseDataConfig):
        self.config = config
        self.base_dir = config.base_dir
    
    def ensure_datasets_exist(self) -> bool:
        """データセットの存在確認と必要に応じた生成"""
        
        # 実際のデータセットファイルの確認
        if self.config.require_all_files:
            real_files_exist = all(
                (self.base_dir / filename).exists()
                for filename in self.config.train_files
            )
        else:
            real_files_exist = any(
                (self.base_dir / filename).exists()
                for filename in self.config.train_files
            )
```

このクラスは学習に必要なデータセットの自動管理を担当している。実際のデータファイルの存在確認、不足時のサンプルデータ自動生成、実行モードに応じた適切なデータセット選択を行う。特に重要な機能として、本番環境では実データの存在を厳格に要求し、開発環境では学習継続性を優先したフォールバック機能を提供している。

## 日本語言語学的処理システム

### JapaneseLinguisticProcessor クラス

```python
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
```

日本語の形態素解析とバリエーション生成を担当する中核クラスである。fugashiライブラリを使用した高精度な形態素解析により、品詞、活用形、読み、基本形などの詳細な言語情報を取得し、これを基にした言語学的に自然なテキストバリエーションを生成する。

### 高度なバリエーション生成機能

```python
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
```

この機能は日本語の複雑な言語現象を活用した学習データ拡張を実現している。動詞の活用形変化（丁寧語と普通形の相互変換）、助詞の自然な省略・変更・追加、形容詞・形容動詞の語尾変化、副詞の感情表現豊かな調整、名詞の敬語化・カジュアル化など、日本語特有の表現バリエーションを自動生成する。

## モデル学習最適化システム

### DeepSeekJapaneseTrainer クラス

```python
class DeepSeekJapaneseTrainer:
    """DeepSeek R1日本語特化トレーナー - AMD MI300X最適化版"""
    
    def __init__(self, config: JapaneseDataConfig):
        self.config = config
        self.linguistic_processor = JapaneseLinguisticProcessor()
        self.dataset_manager = DatasetManager(config)
        
        # ROCm/PyTorch最適化設定
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # ROCm環境での高速化
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

AMD MI300X GPUとROCm 6.1環境に特化した学習最適化を実装するクラスである。CUDA互換レイヤーでのテンソル最適化、メモリ効率化、LoRA（Low-Rank Adaptation）による効率的なファインチューニング、動的バッチサイズ調整など、限られたリソースでの最高性能を引き出す技術を統合している。

### LoRA設定最適化

```python
def auto_detect_target_modules(self, model) -> List[str]:
    """モデルアーキテクチャに基づくLoRA対象モジュール自動検出"""
    logger.info("Auto-detecting LoRA target modules...")
    
    target_modules = []
    
    # モデル内のLinearレイヤーを探索
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 一般的なattentionレイヤー名をチェック
            if any(pattern in name.lower() for pattern in [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
                'query', 'key', 'value', 'output',
                'attention', 'self_attn', 'mlp'
            ]):
                module_name = name.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
```

モデルアーキテクチャの自動解析により、最適なLoRA対象モジュールを動的に決定する機能である。DeepSeekモデルの内部構造を詳細に調査し、アテンション機構とMLP層の中から効果的な学習対象を選択することで、限られたパラメータ数での最大限の学習効果を実現している。

## データ処理パイプライン

### 動的データ拡張システム

```python
def load_and_prepare_datasets(self) -> Dict[str, Dataset]:
    """データセットの読み込みと準備"""
    logger.info("Loading and preparing Japanese datasets...")
    
    # データセット存在確認と必要に応じた生成
    if not self.dataset_manager.ensure_datasets_exist():
        raise RuntimeError("Failed to ensure dataset availability")
    
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
```

実行モードに応じた適応的なデータ処理を実装している。試行モードでは迅速な動作確認のため100サンプル、開発モードでは機能検証のため1000サンプル、本番モードでは全データを対象とした拡張を行う。これにより、開発段階での効率性と本番運用での品質の両立を実現している。

## 継続学習サポート

### 段階的学習機能

```python
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
```

既存の学習済みモデルからの継続学習を支援する機能である。PEFTライブラリを活用して既存のLoRAアダプターを読み込み、段階的な学習を可能にしている。これにより、新しいデータセットでの追加学習や、特定タスクへの特化学習を効率的に実行できる。

## 評価とテストシステム

### 簡易モデルテスト機能

```python
def simple_test(model_path: str, tokenizer):
    """簡易的なモデルテスト"""
    try:
        print("Testing model...")
        
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
```

学習完了後のモデル性能を迅速に評価するテスト機能である。日本語の基本的な挨拶、専門用語への対応、概念説明能力など、多様な角度からモデルの日本語理解能力を検証する。DeepSeekの対話形式に合わせたプロンプト構造を使用し、実際の使用場面に近い条件でのテストを実行している。

## ROCm最適化とパフォーマンス

スクリプト全体にわたってAMD MI300X GPUとROCm 6.1環境での最適化が施されている。PyTorchのCUDA互換レイヤーを活用したテンソル演算の高速化、メモリ分割サイズの最適化、混合精度学習（bfloat16）の活用により、NVIDIA GPU環境と同等の学習性能を実現している。また、動的バッチサイズ調整機能により、利用可能なGPUメモリに応じた最適な学習設定を自動選択する。

## インタラクティブ学習設定

コマンドライン実行時のユーザーインターフェースにより、学習パラメータの柔軟な調整が可能である。エポック数、学習率、出力名などの重要な設定項目について、デフォルト値の提示と対話的な変更機能を提供し、研究目的に応じた詳細な実験設定を支援している。また、自動実行モードも提供し、バッチ処理での効率的な学習実行も可能としている。

このスクリプトは、日本語という複雑な言語の特性を深く理解し、最新の機械学習技術とROCm環境の利点を最大限に活用した、実用的かつ研究価値の高い言語モデル学習システムを実現している。