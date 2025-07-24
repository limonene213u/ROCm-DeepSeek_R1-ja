#!/bin/bash
# DeepSeek R1 BPE解析ツール セットアップスクリプト

echo "DeepSeek R1 BPE解析ツール セットアップを開始します..."

# Python環境の確認
if ! command -v python &> /dev/null; then
    echo "エラー: Pythonがインストールされていません"
    exit 1
fi

# Python バージョンの確認
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python バージョン: $python_version"

# 依存関係のインストール選択
echo ""
echo "インストールタイプを選択してください:"
echo "1) 軽量版 (可視化機能なし、推奨)"
echo "2) フル機能版 (可視化機能付き)"
read -p "選択 (1 or 2): " choice

case $choice in
    1)
        echo "軽量版の依存関係をインストールします..."
        pip install -r requirements-lite.txt
        echo "軽量版のインストールが完了しました！"
        echo "実行コマンド: python analyze_deepseekr1_lite.py"
        ;;
    2)
        echo "フル機能版の依存関係をインストールします..."
        pip install -r requirements.txt
        echo "フル機能版のインストールが完了しました！"
        echo "実行コマンド: python analyze_deepseekr1.py"
        ;;
    *)
        echo "無効な選択です。1 または 2 を選択してください。"
        exit 1
        ;;
esac

echo ""
echo "セットアップが完了しました！"
echo "詳細な使用方法は README.md を参照してください。"
