#!/bin/bash
# PyTorch ROCm 6.1.0 インストールスクリプト
echo "既存PyTorchをアンインストール中..."
pip uninstall torch torchvision torchaudio -y

echo "ROCm 6.1.0版PyTorchをインストール中..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

echo "PyTorch ROCm 6.1.0 インストール完了"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm HIP: {torch.version.hip if hasattr(torch.version, \"hip\") else \"Not Available\"}')"