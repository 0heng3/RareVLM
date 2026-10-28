from rarevlm_paths import PROJECT_ROOT
from vlm_region_preference_probe_v1.common import read, make_batch, evaluate

ROOT = PROJECT_ROOT.parents[1] / 'outputs/vlm_region_adapter_location_v1'
DATA_ROOT = ROOT.parent / 'vlm_region_preference_probe_v1'
CONFIG = PROJECT_ROOT / 'configs/vlm_region_adapter_location_v1.json'
