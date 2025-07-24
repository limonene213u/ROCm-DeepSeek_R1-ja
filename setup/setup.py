# setup.py - DeepSeek R1日本語特化学習システム セットアップ（RunPod ROCm 6.1.0対応）
import os
import sys
import subprocess
import importlib
import importlib.util
import platform
import json
from typing import List, Tuple, Dict, Any


def print_banner() -> None:
    """セットアップバナー表示"""
    print("=" * 65)
    print("  DeepSeek R1 日本語特化学習システム")
    print("  ROCm 6.1.0 + AMD MI300X 最適化版")
    print("=" * 65)


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
    
    # Python バージョンチェック
    if system_info["python_version"] < (3, 10):
        print(f"[ERROR] Python 3.10+ 必要（現在: {sys.version}）")
        system_info["requirements_met"] = False
    else:
        print(f"[OK] Python: {sys.version}")
    
    # メモリチェック（可能な場合）
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1e9
        print(f"[INFO] システムメモリ: {memory_gb:.1f}GB")
        
        if memory_gb < 16:
            system_info["warnings"].append(
                "16GB以上のメモリを推奨（日本語言語モデル学習用）"
            )
    except ImportError:
        system_info["warnings"].append("psutil未インストール - メモリ情報取得不可")
    
    return system_info


def detect_gpu_environment() -> Tuple[bool, str, Dict[str, Any]]:
    """GPU環境自動検出"""
    print("GPU環境検出中...")
    
    gpu_info = {
        "has_gpu": False,
        "gpu_type": "None",
        "gpu_count": 0,
        "total_vram_gb": 0,
        "recommended_model": "deepseek-r1-distill-qwen-1.5b"
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info.update({
                "has_gpu": True,
                "gpu_type": "CUDA",
                "gpu_count": torch.cuda.device_count(),
            })
            
            # GPU詳細情報
            for i in range(gpu_info["gpu_count"]):
                device_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                gpu_info["total_vram_gb"] += vram_gb
                
                print(f"[INFO] GPU {i}: {device_name} ({vram_gb:.1f}GB)")
            
            # 推奨モデル判定（DeepSeek R1用）
            if gpu_info["total_vram_gb"] >= 64:
                gpu_info["recommended_model"] = "deepseek-r1-distill-qwen-7b"
                print("[OK] 7Bモデル学習可能")
            elif gpu_info["total_vram_gb"] >= 16:
                gpu_info["recommended_model"] = "deepseek-r1-distill-qwen-1.5b"
                print("[OK] 1.5Bモデル学習可能")
            else:
                print("[WARNING] GPU VRAM不足 - 設定調整が必要")
        
        # ROCm環境チェック（RunPod ROCm 6.1.0対応）
        elif hasattr(torch, 'cuda') and torch.cuda.is_available():
            gpu_info.update({
                "has_gpu": True,
                "gpu_type": "ROCm",
                "gpu_count": torch.cuda.device_count(),
            })
            
            # ROCm 6.1.0 環境詳細情報
            try:
                for i in range(gpu_info["gpu_count"]):
                    device_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = props.total_memory / 1e9
                    gpu_info["total_vram_gb"] += vram_gb
                    print(f"[INFO] ROCm GPU {i}: {device_name} ({vram_gb:.1f}GB)")
                
                # MI300シリーズの特別扱い
                if "MI300" in str(torch.cuda.get_device_name(0)):
                    gpu_info["recommended_model"] = "deepseek-r1-distill-qwen-7b"
                    print("[OK] MI300シリーズ検出 - 7Bモデル推奨")
                else:
                    gpu_info["recommended_model"] = "deepseek-r1-distill-qwen-7b"
                    
            except Exception:
                # フォールバック処理
                gpu_info["recommended_model"] = "deepseek-r1-distill-qwen-1.5b"
                
            print(f"[INFO] ROCm環境検出 (RunPod ROCm 6.1.0)")
        
        # ROCm環境の場合、torch.cuda.is_available()がTrueでもGPU種別をROCmに変更
        if "gfx" in str(torch.cuda.get_device_properties(0).name).lower():
            if gpu_info["gpu_type"] != "ROCm":
                print("[INFO] CUDA/ROCm混在検出 - ROCm環境として扱います")
                gpu_info["gpu_type"] = "ROCm"
        
        
        else:
            print("[WARNING] GPU未検出 - CPU学習（非常に遅い）")
            
    except ImportError:
        print("[WARNING] PyTorch未インストール")
        return False, "Unknown", gpu_info
    
    return gpu_info["has_gpu"], gpu_info["gpu_type"], gpu_info


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
        
        # ROCm版PyTorchをインストール
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/rocm6.1"
        ]
        print("ROCm 6.1.0版PyTorchをインストール中...")
    else:
        # CUDA/CPU用PyTorch
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ]
        print("CUDA版PyTorchをインストール中...")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("[OK] PyTorch インストール完了")
        
        # インストール確認
        try:
            import torch
            print(f"[OK] PyTorch バージョン: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"[OK] CUDA デバイス利用可能")
            else:
                print("[INFO] CPU版PyTorch")
        except ImportError:
            print("[WARNING] PyTorchインポート確認失敗")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyTorch インストール失敗: {e}")
        return False


def install_requirements(gpu_type: str) -> bool:
    """requirements.txt からパッケージインストール"""
    print("依存パッケージインストール中...")
    
    # bitsandbytes は ROCm環境では除外
    excluded_packages = []
    if gpu_type == "ROCm":
        excluded_packages = ["bitsandbytes"]
        print("🔥 ROCm環境：bitsandbytesをスキップ")
    
    try:
        # requirements.txt を一時的に修正（ROCm環境用）
        with open("requirements.txt", "r") as f:
            requirements = f.readlines()
        
        filtered_requirements = []
        for line in requirements:
            skip = False
            for excluded in excluded_packages:
                if excluded in line and not line.strip().startswith("#"):
                    skip = True
                    break
            if not skip:
                filtered_requirements.append(line)
        
        # 一時ファイル作成
        temp_req_file = "requirements_filtered.txt"
        with open(temp_req_file, "w") as f:
            f.writelines(filtered_requirements)
        
        # インストール実行
        cmd = [sys.executable, "-m", "pip", "install", "-r", temp_req_file]
        subprocess.run(cmd, check=True)
        
        # 一時ファイル削除
        os.remove(temp_req_file)
        
        print("[OK] 依存パッケージインストール完了")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 依存パッケージインストール失敗: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] セットアップエラー: {e}")
        return False


def verify_installation() -> Dict[str, bool]:
    """インストール検証"""
    print("🔍 インストール検証中...")
    
    verification_results = {}
    
    # 必須パッケージ確認
    required_packages = [
        "torch", "transformers", "peft", "datasets", 
        "accelerate", "tqdm", "numpy"
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
            verification_results[package] = True
        except ImportError:
            print(f"[ERROR] {package}")
            verification_results[package] = False
    
    # 特別なチェック
    try:
        import torch
        if torch.cuda.is_available():
            print("[OK] GPU加速利用可能")
            verification_results["gpu_acceleration"] = True
        else:
            print("[WARNING] GPU加速利用不可")
            verification_results["gpu_acceleration"] = False
    except:
        verification_results["gpu_acceleration"] = False
    
    return verification_results


def create_project_structure() -> None:
    """プロジェクト構造作成"""
    print("📁 プロジェクト構造作成中...")
    
    directories = [
        "data",
        "models", 
        "logs",
        "results",
        "deployment",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 {directory}/")


def create_sample_files() -> None:
    """サンプルファイル作成"""
    print("📝 サンプルファイル作成中...")
    
    # サンプル設定ファイル
    sample_config = {
        "character_name": "akatsuki",
        "model_name": "nekomata-14b",
        "training_config": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 5
        }
    }
    
    with open("config_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    # サンプルデータセット
    sample_dataset = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "あなたはDeepSeek R1の日本語特化アシスタントです。"
                },
                {
                    "role": "user",
                    "content": "こんにちは"
                },
                {
                    "role": "assistant",
                    "content": "こんにちは！私はDeepSeek R1の日本語アシスタントです。何かお手伝いできることはありますか？"
                }
            ]
        }
    ]
    
    with open("data/sample_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in sample_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("📝 サンプルファイル作成完了")


def generate_quick_start_guide(gpu_info: Dict[str, Any]) -> None:
    """クイックスタートガイド生成（RunPod ROCm 6.1.0対応）"""
    # RunPod環境の検出
    runpod_env = ""
    if gpu_info.get('gpu_type', '') == 'ROCm':
        runpod_env = "- 環境: RunPod PyTorch 2.4.0-py3.10-rocm6.1.0-ubuntu22.04\n"
    
    guide_content = f"""# DeepSeek R1 日本語特化学習 クイックスタートガイド

## 🎯 検出された環境
- GPU: {gpu_info.get('gpu_type', 'None')}
- VRAM: {gpu_info.get('total_vram_gb', 0):.1f}GB
- 推奨モデル: {gpu_info.get('recommended_model', 'nekomata-7b')}
{runpod_env}

## 📋 基本実行手順

### 1. テスト実行（推奨：最初に実行）
```bash
# 依存関係確認
python main.py --check-only

# 7Bモデルでテスト（約30分、安価）
python main.py --preset akatsuki --model nekomata-7b
```

### 2. 本格学習
```bash
# 推奨モデルで本格学習
python main.py --preset akatsuki --model {gpu_info.get('recommended_model', 'nekomata-7b')}
```

### 3. カスタム設定
```bash
# 設定ファイル編集後
python main.py --config config_sample.json
```

## 📝 RunPod ROCm 6.1.0 環境での注意事項
- PyTorch 2.4.0 + ROCm 6.1.0 の最新環境に最適化済み
- MI300シリーズ GPU で最高のパフォーマンスを発揮
- 初回実行時はモデルダウンロードで時間がかかります
- GPU VRAM不足の場合は batch_size を調整してください
- 学習ログは logs/ フォルダに保存されます

## 🆘 トラブルシューティング
- CUDA Out of Memory → batch_size を減らす
- ModuleNotFoundError → pip install -r requirements.txt
- GPU未検出 → ドライバー確認
- ROCm関連エラー → ROCm 6.1.0 ドライバー確認

詳細はDocumentation/フォルダ内のガイドを参照してください。
"""
    
    with open("QUICKSTART.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print("📝 クイックスタートガイド生成完了: QUICKSTART.md")


def main() -> None:
    """メイン実行関数"""
    try:
        print_banner()
        
        # システム要件チェック
        system_info = check_system_requirements()
        if not system_info["requirements_met"]:
            print("[ERROR] システム要件を満たしていません")
            sys.exit(1)
        
        # 警告表示
        for warning in system_info["warnings"]:
            print(f"[WARNING] {warning}")
        
        # GPU環境検出
        has_gpu, gpu_type, gpu_info = detect_gpu_environment()
        
        # プロジェクト構造作成
        create_project_structure()
        
        # PyTorchインストール確認と強制ROCm対応
        force_rocm = False
        try:
            import torch
            current_version = torch.__version__
            is_rocm = "rocm" in current_version.lower()
            
            print(f"[OK] PyTorch既にインストール済み: {current_version}")
            
            # CUDA版が入っている場合、ROCm環境では強制的に切り替え
            if not is_rocm and gpu_type == "ROCm":
                print("[WARNING] CUDA版PyTorchが検出されました。ROCm版に切り替えます...")
                force_rocm = True
            elif is_rocm:
                print(f"[OK] ROCm版PyTorch確認済み")
                
        except ImportError:
            print("[WARNING] PyTorch未インストール")
            force_rocm = (gpu_type == "ROCm")
        
        # PyTorchインストール/再インストール
        if force_rocm or not importlib.util.find_spec("torch"):
            if not install_pytorch(gpu_type):
                print("[ERROR] PyTorchインストール失敗")
                sys.exit(1)
        
        # 依存パッケージインストール
        if not install_requirements(gpu_type):
            print("[ERROR] 依存パッケージインストール失敗")
            sys.exit(1)
        
        # インストール検証
        verification_results = verify_installation()
        failed_packages = [pkg for pkg, success in verification_results.items() if not success]
        
        if failed_packages:
            print(f"[WARNING] 一部パッケージのインストールに失敗: {failed_packages}")
        
        # サンプルファイル作成
        create_sample_files()
        
        # クイックスタートガイド生成
        generate_quick_start_guide(gpu_info)
        
        # 完了メッセージ
        print("\n" + "=" * 50)
        print("[OK] セットアップ完了！")
        print("QUICKSTART.md を確認して学習を開始してください")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n[ERROR] セットアップが中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()