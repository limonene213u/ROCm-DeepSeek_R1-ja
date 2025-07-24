# setup.py コード詳細説明

## 概要

setup.pyは、DeepSeek R1日本語特化学習システムの自動環境構築スクリプトである。RunPod ROCm 6.1.0環境とAMD MI300X GPUに最適化された設定で、PyTorchやHuggingFace transformersなどの依存関係を自動インストールし、日本語言語モデル学習環境を一括構築する。

## 実装意図

このスクリプトは、異なる実行環境（開発環境、クラウド環境）での環境構築の一貫性を保つことを主目的としている。特にROCm環境とCUDA環境の自動判別、適切なPyTorchバージョンの選択、GPU最適化設定の自動適用により、研究の再現性を確保する。また、セットアップエラーを最小限に抑制し、初学者でも容易に学習環境を構築できるよう設計されている。

## システム要件チェック機能

### check_system_requirements 関数

```python
def check_system_requirements() -> Dict[str, Any]:
    """システム要件チェック"""
    print("システム要件チェック中...")
    
    system_info = {
        "python_version": sys.version_info,
        "platform": platform.system(),
        "architecture": platform.architecture()[0],
        "cpu_count": os.cpu_count(),
        "requirements_met": True,
        "warnings": []
    }
```

この関数はセットアップ実行前の環境検証を行う。Python バージョン、プラットフォーム情報、アーキテクチャ、CPU数を取得し、システム要件との適合性を判定する。要件を満たさない場合は処理を停止し、警告が必要な場合は適切なメッセージを表示する。

### GPU 環境自動検出

```python
def detect_gpu_environment() -> str:
    """GPU環境自動検出（ROCm/CUDA/CPU）"""
    print("GPU環境検出中...")
    
    # ROCm チェック（優先）
    if os.path.exists('/opt/rocm') or 'ROCM_HOME' in os.environ:
        print("[INFO] ROCm環境検出 (RunPod ROCm 6.1.0)")
        return "ROCm"
```

GPU環境の自動判別機能を提供している。ROCmインストールの存在確認、環境変数の検査、PyTorchのGPU利用可能性チェックを順次実行し、最適なインストール設定を決定する。RunPod環境での実行を前提とし、ROCm 6.1.0の検出を最優先としている。

## PyTorch 自動インストール機能

### install_pytorch 関数

```python
def install_pytorch(gpu_type: str) -> bool:
    """GPU環境に応じたPyTorch自動インストール"""
    print(f"PyTorch インストール中（{gpu_type}環境用）...")
    
    if gpu_type == "ROCm":
        # ROCm用PyTorch（RunPod ROCm 6.1.0環境対応）
        # 既存PyTorchをアンインストール
        print("既存PyTorchをアンインストール中...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", 
                "torch", "torchvision", "torchaudio", "-y"
            ], check=False, capture_output=True)  # エラーを無視
        except:
            pass
```

GPU環境に応じて適切なPyTorchバージョンを自動選択・インストールする。ROCm環境では専用のindex URLから取得し、CUDA環境では標準版を使用する。既存インストールとの競合を避けるため、強制的なアンインストール機能も実装している。

## 依存関係管理

### install_dependencies 関数

```python
def install_dependencies() -> bool:
    """依存パッケージの一括インストール"""
    print("依存パッケージインストール中...")
    
    requirements = [
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "safetensors>=0.4.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "japanize-matplotlib>=1.1.3",
        "fugashi>=1.3.0",
        "ipadic>=1.0.0",
        "mecab-python3>=1.0.6",
        "psutil>=5.9.0",
        "packaging>=23.0"
    ]
```

DeepSeek R1日本語学習に必要な全依存関係を定義し、バージョン指定による互換性確保を行っている。HuggingFace ecosystem、日本語処理ライブラリ、可視化ツール、システム監視ツールなど、包括的なパッケージセットを提供している。

## GPU 最適化設定

### get_gpu_info 関数

```python
def get_gpu_info() -> Dict[str, Any]:
    """詳細なGPU情報取得（ROCm/CUDA対応）"""
    gpu_info = {
        "available": False,
        "type": "CPU",
        "device_count": 0,
        "devices": [],
        "memory_info": {},
        "recommended_models": []
    }
```

利用可能なGPUの詳細情報を取得し、推奨学習設定を提案する機能を提供している。AMD MI300XやNVIDIA GPU の性能に応じて、適切なモデルサイズやバッチサイズを自動提案し、効率的な学習環境設定を支援している。

## サンプルデータ生成

### generate_sample_data 関数

```python
def generate_sample_data() -> None:
    """学習用サンプルデータ生成"""
    print("サンプルデータ生成中...")
    
    # データセットディレクトリ作成
    dataset_dir = Path("../dataset/deepseek-jp")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # サンプル会話データ
    sample_conversations = [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "DeepSeek R1について教えてください。"
                },
                {
                    "from": "assistant",
                    "content": "あなたはDeepSeek R1の日本語特化アシスタントです。"
                }
```

学習用のサンプルデータセットを自動生成する機能を実装している。DeepSeek R1の日本語特化学習に適した対話形式のデータを生成し、初期学習や動作確認に使用できる。生成されるデータは実際の学習フォーマットと同一構造を持ち、パイプラインの検証に活用される。

## クイックスタートガイド生成

### generate_quick_start_guide 関数

```python
def generate_quick_start_guide(gpu_info: Dict[str, Any]) -> None:
    """クイックスタートガイド生成"""
    print("クイックスタートガイド生成中...")
    
    guide_content = f"""# DeepSeek R1 日本語特化学習 クイックスタートガイド

## 環境情報
- GPU: {gpu_info.get('type', 'CPU')}
- デバイス数: {gpu_info.get('device_count', 0)}
- 推奨モデル: {', '.join(gpu_info.get('recommended_models', ['DeepSeek-R1-1.5B']))}
```

セットアップ完了後に、現在の環境設定に合わせたクイックスタートガイドを自動生成する。検出されたGPU情報、推奨設定、実行コマンド例を含む包括的な使用説明書を提供し、ユーザーの学習開始を支援している。

## エラーハンドリングと検証

### verify_installation 関数

```python
def verify_installation() -> bool:
    """インストール結果の検証"""
    print("インストール検証中...")
    
    required_packages = [
        "torch", "transformers", "datasets", "tokenizers",
        "accelerate", "peft", "numpy", "pandas"
    ]
    
    failed_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[ERROR] {package}")
            failed_packages.append(package)
```

インストール完了後の動作確認を自動実行する。必須パッケージのインポートテスト、GPU利用可能性の確認、基本機能の動作検証を行い、セットアップの成功を保証している。問題が発見された場合は具体的なエラー情報を提供し、トラブルシューティングを支援する。

## メイン実行フロー

```python
def main():
    """メイン処理"""
    try:
        print_banner()
        
        # システム要件チェック
        system_info = check_system_requirements()
        if not system_info["requirements_met"]:
            print("[ERROR] システム要件を満たしていません")
            sys.exit(1)
        
        if system_info["warnings"]:
            for warning in system_info["warnings"]:
                print(f"[WARNING] {warning}")
```

スクリプト全体の実行制御を管理している。システム要件チェック、GPU環境検出、PyTorchインストール、依存関係インストール、動作確認、ガイド生成の各段階を順次実行し、エラー発生時の適切な終了処理を提供している。

## ROCm 最適化

スクリプト全体にわたってROCm 6.1.0環境とAMD MI300X GPUに特化した最適化を実装している。ROCm特有のPyTorchインデックスURL、メモリ管理設定、パフォーマンスチューニングなど、AMD GPU環境での最高性能を引き出すための設定を自動適用している。これにより、NVIDIA GPU環境と同等の学習効率を実現し、ROCm環境での日本語言語モデル研究を支援している。
