"""Reproducible data pipeline (fixed seed).   python make_data.py [--train] [--dry-run]

Steps (all scripts are Member 1's, run in this order from the project root):
  1 prepare_targets   FFF-module TMA scores            -> data/processed/fff_tma_targets.csv
  2 create_features   previous-TMA features            -> data/processed/fff_tma_features.csv
  3 add_vle_features  clicks / active days / resources -> data/processed/fff_tma_features.csv
  4 prepare_splits    train 2013B+2013J, val 2014B, test 2014J
  5 scale_features    StandardScaler fitted on train only -> *_scaled.csv + models/scaler.pkl
  --train  6 train_model -> models/study_planner_nn.pth (needs PyTorch), then export numpy weights

Raw OULAD CSVs are NOT in the repo. Download OULAD (CC BY 4.0) from
https://analyse.kmi.open.ac.uk/open_dataset and unzip into data/raw/oulad/
(needs assessments.csv, studentAssessment.csv, studentVle.csv).
"""
import argparse
import os
import random
import runpy
import sys
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "oulad"
REQUIRED_RAW = ["assessments.csv", "studentAssessment.csv", "studentVle.csv"]
STEPS = ["prepare_targets", "create_features", "add_vle_features", "prepare_splits", "scale_features"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="also retrain the NN (needs torch)")
    ap.add_argument("--dry-run", action="store_true", help="only list the steps")
    args = ap.parse_args()
    steps = STEPS + (["train_model"] if args.train else [])
    if args.dry_run:
        print("Steps:", " -> ".join(steps)); return 0
    missing = [f for f in REQUIRED_RAW if not (RAW / f).exists()]
    if missing:
        print(f"ERROR: raw OULAD files missing in {RAW}: {missing}\n"
              "Download OULAD from https://analyse.kmi.open.ac.uk/open_dataset and unzip it there.")
        return 1
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
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
