
"""Reproducible data pipeline (fixed seed).   python make_data.py [--train] [--dry-run]

Steps (all scripts are Member 1's, run in this order from the project root):
  1 prepare_targets   FFF-module TMA scores            -> data/processed/fff_tma_targets.csv
  2 create_features   previous-TMA features            -> data/processed/fff_tma_features.csv
  3 add_vle_features  clicks / active days / resources -> data/processed/fff_tma_features.csv
  4 prepare_splits    train 2013B+2013J, val 2014B, test 2014J
  5 scale_features    StandardScaler fitted on train only -> *_scaled.csv + models/scaler.pkl
  --train  6 train_model -> models/study_planner_nn.pth (needs PyTorch), then export numpy weights

Raw OULAD CSVs are downloaded automatically if missing.
"""

import argparse
import importlib
import os
import random
import runpy
import sys
import shutil
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path


SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "oulad"
REQUIRED_RAW = ["assessments.csv", "studentAssessment.csv", "studentVle.csv"]
STEPS = ["prepare_targets", "create_features", "add_vle_features", "prepare_splits", "scale_features"]

# Direct OULAD dataset download
DATASET_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"


def download_dataset():
    RAW.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED_RAW if not (RAW / f).exists()]

    if not missing:
        print("All required OULAD files already exist.")
        return True

    print("Downloading missing OULAD files:", missing)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "oulad.zip"

            request = urllib.request.Request(
                DATASET_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(request, timeout=120) as response:
                with open(zip_path, "wb") as output:
                    shutil.copyfileobj(response, output)

            with zipfile.ZipFile(zip_path, "r") as archive:
                available = {
                    Path(name).name: name
                    for name in archive.namelist()
                    if Path(name).name in missing
                }

                if len(available) != len(missing):
                    print("ERROR: Some required files were not found in the dataset ZIP.")
                    return False

                for filename in missing:
                    with archive.open(available[filename]) as source:
                        with open(RAW / filename, "wb") as destination:
                            shutil.copyfileobj(source, destination)

                    print(f"Downloaded: {filename}")

        print("OULAD dataset is ready.")
        return True

    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as e:
        print(f"ERROR: Could not download OULAD dataset: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="also retrain the NN (needs torch)")
    ap.add_argument("--dry-run", action="store_true", help="only list the steps")
    args = ap.parse_args()
    steps = STEPS + (["train_model"] if args.train else [])
    if args.dry_run:
        print("Steps:", " -> ".join(steps)); return 0

    if not download_dataset():
        return 1

    random.seed(SEED)
    try:
        importlib.import_module("numpy").random.seed(SEED)
    except ImportError:
        pass
    os.chdir(ROOT)                                   # the scripts use project-root-relative paths
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / "src" / "nn"))     # train_model.py does "from model import ..."
    for name in steps:
        print(f"\n### {name}")
        runpy.run_path(str(ROOT / "src" / "nn" / f"{name}.py"), run_name="__main__")
    if args.train:
        runpy.run_path(str(ROOT / "src" / "nn" / "export_numpy_weights.py"), run_name="__main__")
    print("\nData ready in data/processed/. Check data/processed/test.csv exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
