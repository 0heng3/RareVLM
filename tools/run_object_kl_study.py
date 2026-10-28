#!/usr/bin/env python3
"""Persistent KL study entry; --prepare-only validates/locks without GPU."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_object_kl_study.run import main

if __name__ == "__main__":
    main()
