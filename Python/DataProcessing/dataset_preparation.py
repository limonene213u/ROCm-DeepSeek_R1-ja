"""
データセット準備・管理モジュール
JGLUE、Japanese MT-Bench、llm-jp-eval の自動ダウンロード・前処理
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request
import zipfile
import tarfile

try:
    import datasets
    from datasets import Dataset, DatasetDict
    import requests
except ImportError:
    print("Warning: datasets, requests not available. Install with: pip install datasets requests")

class DatasetManager:
    """データセット管理システム"""
    
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        self.logger = logging.getLogger("DatasetManager")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def prepare_jglue_datasets(self) -> Dict[str, Any]:
        """JGLUE データセットの準備"""
        self.logger.info("🔄 Preparing JGLUE datasets")
        
        jglue_dir = self.output_dir / "jglue"
        jglue_dir.mkdir(exist_ok=True)
        
        # JGLUE タスク一覧
        jglue_tasks = [
            "MARC-ja",      # Multi-domain Amazon Review Classification
            "JSTS",         # Japanese Semantic Textual Similarity
            "JNLI",         # Japanese Natural Language Inference
            "JSQuAD",       # Japanese Stanford Question Answering Dataset
            "JCommonsenseQA", # Japanese Commonsense Question Answering
            "NIILC",        # Natural Instructions in Japanese
        ]
        
        results = {
            "status": "PREPARING",
            "tasks_prepared": [],
            "errors": []
        }
        
        try:
            # datasets ライブラリを使用したダウンロード
            for task in jglue_tasks:
                try:
                    self.logger.info(f"Downloading {task}...")
                    
                    # Hugging Face datasets から JGLUE を取得
                    if task == "MARC-ja":
                        dataset = datasets.load_dataset("shunk031/JGLUE", "MARC-ja", cache_dir=str(jglue_dir))
                    elif task == "JSTS":
                        dataset = datasets.load_dataset("shunk031/JGLUE", "JSTS", cache_dir=str(jglue_dir))
                    elif task == "JNLI":
                        dataset = datasets.load_dataset("shunk031/JGLUE", "JNLI", cache_dir=str(jglue_dir))
                    elif task == "JSQuAD":
                        dataset = datasets.load_dataset("shunk031/JGLUE", "JSQuAD", cache_dir=str(jglue_dir))
                    elif task == "JCommonsenseQA":
                        dataset = datasets.load_dataset("shunk031/JGLUE", "JCommonsenseQA", cache_dir=str(jglue_dir))
                    else:
                        # フォールバック：GitHubから直接取得
                        self._download_jglue_from_github(task, jglue_dir)
                        continue
                    
                    # Parquet形式で保存
                    task_dir = jglue_dir / task.lower()
                    task_dir.mkdir(exist_ok=True)
                    
                    for split_name, split_data in dataset.items():
                        parquet_file = task_dir / f"{split_name}.parquet"
                        split_data.to_parquet(str(parquet_file))
                        self.logger.info(f"Saved {task} {split_name}: {len(split_data)} examples")
                    
                    results["tasks_prepared"].append(task)
                    
                except Exception as e:
                    error_msg = f"Failed to prepare {task}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            results["status"] = "COMPLETED" if len(results["tasks_prepared"]) > 0 else "FAILED"
            
        except Exception as e:
            error_msg = f"JGLUE preparation failed: {e}"
            self.logger.error(error_msg)
            results["status"] = "FAILED"
            results["errors"].append(error_msg)
        
        return results
    
    def _download_jglue_from_github(self, task: str, output_dir: Path):
        """GitHub から JGLUE を直接ダウンロード（フォールバック）"""
        try:
            github_url = "https://github.com/yahoojapan/JGLUE/archive/refs/heads/main.zip"
            zip_file = output_dir / "jglue_main.zip"
            
            # ダウンロード
            if not zip_file.exists():
                urllib.request.urlretrieve(github_url, zip_file)
            
            # 解凍
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            self.logger.info(f"Downloaded JGLUE from GitHub for task: {task}")
            
        except Exception as e:
            self.logger.error(f"GitHub download failed for {task}: {e}")
    
    def prepare_japanese_mtbench(self) -> Dict[str, Any]:
        """Japanese MT-Bench データセットの準備"""
        self.logger.info("🔄 Preparing Japanese MT-Bench dataset")
        
        mtbench_dir = self.output_dir / "japanese_mt_bench"
        mtbench_dir.mkdir(exist_ok=True)
        
        results = {
            "status": "PREPARING",
            "datasets_prepared": [],
            "errors": []
        }
        
        try:
            # Hugging Face からダウンロード
            dataset = datasets.load_dataset(
                "naive-puzzle/japanese-mt-bench", 
                cache_dir=str(mtbench_dir)
            )
            
            # Parquet形式で保存
            for split_name, split_data in dataset.items():
                parquet_file = mtbench_dir / f"{split_name}.parquet"
                split_data.to_parquet(str(parquet_file))
                self.logger.info(f"Saved Japanese MT-Bench {split_name}: {len(split_data)} examples")
                results["datasets_prepared"].append(f"mt_bench_{split_name}")
            
            # メタデータ保存
            metadata = {
                "dataset_name": "japanese-mt-bench",
                "source": "naive-puzzle/japanese-mt-bench",
                "download_date": self._get_current_datetime(),
                "task_type": "generation_evaluation",
                "language": "ja"
            }
            
            with open(mtbench_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            results["status"] = "COMPLETED"
            
        except Exception as e:
            error_msg = f"Japanese MT-Bench preparation failed: {e}"
            self.logger.error(error_msg)
            results["status"] = "FAILED" 
            results["errors"].append(error_msg)
        
        return results
    
    def prepare_llmjp_eval(self) -> Dict[str, Any]:
        """llm-jp-eval データセットの準備"""
        self.logger.info("🔄 Preparing llm-jp-eval datasets")
        
        llmjp_dir = self.output_dir / "llm_jp_eval"
        llmjp_dir.mkdir(exist_ok=True)
        
        results = {
            "status": "PREPARING",
            "tasks_prepared": [],
            "errors": []
        }
        
        # llm-jp-eval のタスク一覧
        llmjp_tasks = [
            "jcommonsenseqa",
            "jemhopqa",
            "jsquad",
            "jaqket_v2",
            "xlsum_ja",
            "mgsm",
            "jcola",
            "jblimp",
            "chabsa",
            "niilc",
            "janli",
            "wikipedia"
        ]
        
        try:
            for task in llmjp_tasks:
                try:
                    # まずは主要タスクのみHugging Faceから取得を試行
                    if task in ["jcommonsenseqa", "jsquad"]:
                        self.logger.info(f"Downloading {task} from Hugging Face...")
                        dataset = datasets.load_dataset(f"llm-jp/{task}", cache_dir=str(llmjp_dir))
                        
                        # Parquet保存
                        task_dir = llmjp_dir / task
                        task_dir.mkdir(exist_ok=True)
                        
                        for split_name, split_data in dataset.items():
                            parquet_file = task_dir / f"{split_name}.parquet"
                            split_data.to_parquet(str(parquet_file))
                            self.logger.info(f"Saved {task} {split_name}: {len(split_data)} examples")
                        
                        results["tasks_prepared"].append(task)
                    else:
                        # その他は後で追加実装
                        self.logger.info(f"Task {task} will be implemented later")
                        
                except Exception as e:
                    error_msg = f"Failed to prepare {task}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            results["status"] = "COMPLETED" if len(results["tasks_prepared"]) > 0 else "PARTIAL"
            
        except Exception as e:
            error_msg = f"llm-jp-eval preparation failed: {e}"
            self.logger.error(error_msg)
            results["status"] = "FAILED"
            results["errors"].append(error_msg)
        
        return results
    
    def _get_current_datetime(self) -> str:
        """現在時刻を取得"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def verify_dataset_integrity(self) -> Dict[str, Any]:
        """データセット整合性チェック"""
        self.logger.info("🔍 Verifying dataset integrity")
        
        verification_results = {
            "jglue": self._verify_jglue_integrity(),
            "mt_bench": self._verify_mtbench_integrity(),
            "llm_jp_eval": self._verify_llmjp_integrity()
        }
        
        return verification_results
    
    def _verify_jglue_integrity(self) -> Dict[str, Any]:
        """JGLUE データセット整合性チェック"""
        jglue_dir = self.output_dir / "jglue"
        
        if not jglue_dir.exists():
            return {"status": "NOT_FOUND", "message": "JGLUE directory not found"}
        
        expected_tasks = ["marc-ja", "jsts", "jnli", "jsquad", "jcommonsenseqa"]
        found_tasks = []
        
        for task in expected_tasks:
            task_dir = jglue_dir / task
            if task_dir.exists() and any(task_dir.glob("*.parquet")):
                found_tasks.append(task)
        
        return {
            "status": "VERIFIED" if len(found_tasks) > 0 else "INCOMPLETE",
            "found_tasks": found_tasks,
            "expected_tasks": expected_tasks,
            "completion_rate": f"{len(found_tasks)}/{len(expected_tasks)}"
        }
    
    def _verify_mtbench_integrity(self) -> Dict[str, Any]:
        """MT-Bench データセット整合性チェック"""
        mtbench_dir = self.output_dir / "japanese_mt_bench"
        
        if not mtbench_dir.exists():
            return {"status": "NOT_FOUND", "message": "Japanese MT-Bench directory not found"}
        
        parquet_files = list(mtbench_dir.glob("*.parquet"))
        metadata_file = mtbench_dir / "metadata.json"
        
        return {
            "status": "VERIFIED" if len(parquet_files) > 0 else "INCOMPLETE",
            "parquet_files": len(parquet_files),
            "has_metadata": metadata_file.exists()
        }
    
    def _verify_llmjp_integrity(self) -> Dict[str, Any]:
        """llm-jp-eval データセット整合性チェック"""
        llmjp_dir = self.output_dir / "llm_jp_eval"
        
        if not llmjp_dir.exists():
            return {"status": "NOT_FOUND", "message": "llm-jp-eval directory not found"}
        
        task_dirs = [d for d in llmjp_dir.iterdir() if d.is_dir()]
        
        return {
            "status": "VERIFIED" if len(task_dirs) > 0 else "INCOMPLETE",
            "task_directories": len(task_dirs),
            "tasks": [d.name for d in task_dirs]
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """データセット統計情報取得"""
        self.logger.info("📊 Collecting dataset statistics")
        
        stats = {
            "total_datasets": 0,
            "total_examples": 0,
            "dataset_breakdown": {}
        }
        
        # JGLUE統計
        jglue_stats = self._get_jglue_stats()
        stats["dataset_breakdown"]["jglue"] = jglue_stats
        stats["total_examples"] += jglue_stats.get("total_examples", 0)
        
        # MT-Bench統計
        mtbench_stats = self._get_mtbench_stats()
        stats["dataset_breakdown"]["mt_bench"] = mtbench_stats
        stats["total_examples"] += mtbench_stats.get("total_examples", 0)
        
        # llm-jp-eval統計
        llmjp_stats = self._get_llmjp_stats()
        stats["dataset_breakdown"]["llm_jp_eval"] = llmjp_stats
        stats["total_examples"] += llmjp_stats.get("total_examples", 0)
        
        stats["total_datasets"] = len([k for k, v in stats["dataset_breakdown"].items() if v.get("total_examples", 0) > 0])
        
        return stats
    
    def _get_jglue_stats(self) -> Dict[str, Any]:
        """JGLUE統計取得"""
        jglue_dir = self.output_dir / "jglue"
        
        if not jglue_dir.exists():
            return {"total_examples": 0, "tasks": {}}
        
        stats = {"total_examples": 0, "tasks": {}}
        
        for task_dir in jglue_dir.iterdir():
            if task_dir.is_dir():
                task_examples = 0
                parquet_files = list(task_dir.glob("*.parquet"))
                
                for parquet_file in parquet_files:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(parquet_file)
                        task_examples += len(df)
                    except:
                        pass
                
                stats["tasks"][task_dir.name] = task_examples
                stats["total_examples"] += task_examples
        
        return stats
    
    def _get_mtbench_stats(self) -> Dict[str, Any]:
        """MT-Bench統計取得"""
        mtbench_dir = self.output_dir / "japanese_mt_bench"
        
        if not mtbench_dir.exists():
            return {"total_examples": 0}
        
        total_examples = 0
        parquet_files = list(mtbench_dir.glob("*.parquet"))
        
        for parquet_file in parquet_files:
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_file)
                total_examples += len(df)
            except:
                pass
        
        return {"total_examples": total_examples, "files": len(parquet_files)}
    
    def _get_llmjp_stats(self) -> Dict[str, Any]:
        """llm-jp-eval統計取得"""
        llmjp_dir = self.output_dir / "llm_jp_eval"
        
        if not llmjp_dir.exists():
            return {"total_examples": 0, "tasks": {}}
        
        stats = {"total_examples": 0, "tasks": {}}
        
        for task_dir in llmjp_dir.iterdir():
            if task_dir.is_dir():
                task_examples = 0
                parquet_files = list(task_dir.glob("*.parquet"))
                
                for parquet_file in parquet_files:
                    try:
                        import pandas as pd
                        df = pd.read_parquet(parquet_file)
                        task_examples += len(df)
                    except:
                        pass
                
                stats["tasks"][task_dir.name] = task_examples
                stats["total_examples"] += task_examples
        
        return stats


# データセット準備のヘルパー関数
def setup_evaluation_datasets(output_dir: str = "datasets") -> Dict[str, Any]:
    """評価用データセット一括セットアップ"""
    manager = DatasetManager(output_dir)
    
    results = {
        "jglue": manager.prepare_jglue_datasets(),
        "mt_bench": manager.prepare_japanese_mtbench(),
        "llm_jp_eval": manager.prepare_llmjp_eval()
    }
    
    # 整合性チェック
    verification = manager.verify_dataset_integrity()
    results["verification"] = verification
    
    # 統計情報
    statistics = manager.get_dataset_statistics()
    results["statistics"] = statistics
    
    return results


if __name__ == "__main__":
    # テスト実行
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "test_datasets"
    
    print("🔄 Starting dataset preparation...")
    results = setup_evaluation_datasets(output_dir)
    
    print("\n📊 Dataset Preparation Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
