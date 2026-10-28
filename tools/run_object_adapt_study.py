#!/usr/bin/env python3
"""Run the persistent object adaptation study (explicit --prepare-only for CPU)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_object_study.run import main

if __name__ == "__main__":
    main()
