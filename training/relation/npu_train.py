"""Explicit Ascend relation-gated feature training; torch_npu loads on device use."""
from pathlib import Path
from training.temporal.cli import main

if __name__ == '__main__':
    raise SystemExit(main('npu',default_config=Path(__file__).resolve().parents[2]/'configs/training/relation_gated.json'))
