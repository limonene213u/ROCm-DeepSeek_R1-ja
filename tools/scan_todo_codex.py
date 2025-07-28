#!/usr/bin/env python3
"""
TODO/Copilot指示自動抽出ツール

AGENTS.mdの統合運用フローに従って、コードベースからTODO/Copilot指示を抽出し、
TODO.mdファイルを自動生成します。
"""

import os
import sys
from pathlib import Path

# プロジェクトルートディレクトリ設定
PROJECT_ROOT = Path(__file__).parent.parent
TARGET_DIRS = ["Python", "R", "Docs"]
OUTPUT_FILE = PROJECT_ROOT / "TODO_EXTRACTED.md"
KEYWORDS = ["TODO:", "Copilot:"]

def extract_todos():
    """TODOとCopilot指示をコードベースから抽出"""
    results = []
    
    for target_dir in TARGET_DIRS:
        target_path = PROJECT_ROOT / target_dir
        
        if not target_path.exists():
            continue
            
        for root, _, files in os.walk(target_path):
            for file in files:
                # Python, R, Markdown ファイルを対象
                if file.endswith((".py", ".r", ".R", ".md")):
                    full_path = Path(root) / file
                    relative_path = full_path.relative_to(PROJECT_ROOT)
                    
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        
                        for i, line in enumerate(lines):
                            for kw in KEYWORDS:
                                if kw in line:
                                    results.append({
                                        "file": str(relative_path),
                                        "line": i + 1,
                                        "content": line.strip(),
                                        "category": categorize_todo(line, kw)
                                    })
                    except UnicodeDecodeError:
                        print(f"Warning: Could not read {relative_path} due to encoding issues")
                        continue
    
    return results

def categorize_todo(line, keyword):
    """TODO項目をカテゴリ分類"""
    line_lower = line.lower()
    
    if "r-" in line_lower or "validation" in line_lower:
        return "Paper Validation"
    elif "benchmark" in line_lower or "measurement" in line_lower:
        return "Benchmarking"
    elif "implementation" in line_lower or "implement" in line_lower:
        return "Implementation"
    elif "statistical" in line_lower or "confidence" in line_lower:
        return "Statistical Analysis"
    elif "rocm" in line_lower or "mi300x" in line_lower:
        return "Hardware Optimization"
    else:
        return "General"

def write_md(results):
    """抽出結果をMarkdown形式で出力"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# TODO items extracted from codebase\n\n")
        f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Based on AGENTS.md integrated workflow for TODO/Copilot instruction management.\n\n")
        
        # カテゴリ別に整理
        categories = {}
        for item in results:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # 優先度順でカテゴリ表示
        priority_order = [
            "Paper Validation", 
            "Benchmarking", 
            "Implementation", 
            "Statistical Analysis", 
            "Hardware Optimization", 
            "General"
        ]
        
        for category in priority_order:
            if category in categories:
                f.write(f"## {category}\n\n")
                for item in categories[category]:
                    f.write(f"- [ ] **{item['file']}**, line {item['line']}\n")
                    f.write(f"      → `{item['content']}`\n\n")
        
        # 統計情報
        f.write(f"\n## Summary\n\n")
        f.write(f"Total TODO/Copilot instructions found: {len(results)}\n\n")
        for category, items in categories.items():
            f.write(f"- {category}: {len(items)} items\n")

if __name__ == "__main__":
    print("Scanning codebase for TODO and Copilot instructions...")
    todos = extract_todos()
    
    if todos:
        write_md(todos)
        print(f"Extracted {len(todos)} TODO/Copilot instructions into {OUTPUT_FILE}")
        
        # カテゴリ別統計表示
        categories = {}
        for item in todos:
            cat = item["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nCategory breakdown:")
        for category, count in categories.items():
            print(f"  {category}: {count} items")
    else:
        print("No TODO or Copilot instructions found in codebase.")
