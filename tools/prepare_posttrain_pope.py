"""Pin original POPE annotations; optionally fetch one 85MB mirror image shard."""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vlm_posttrain_pope.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download-mirror", action="store_true")
    parser.add_argument("--mirror-parquet", type=Path)
    args = parser.parse_args()
    result = prepare(args.data_root, download_mirror=args.download_mirror, mirror_parquet=args.mirror_parquet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
