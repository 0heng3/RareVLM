import json
from pathlib import Path
from rarevlm_paths import PROJECT_ROOT
from vlm_region_preference_probe_v1.common import make_batch, evaluate

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_decision_sft_v1'
DATA_ROOT = ROOT.parent / 'vlm_region_preference_probe_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_region_decision_sft_v1.json'


def read(path):
    return json.loads(Path(path).read_text())
