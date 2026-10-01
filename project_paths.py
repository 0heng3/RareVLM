"""Portable paths; explicit CLI paths take precedence over these defaults."""
import os
from pathlib import Path

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parent
B32_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
B16_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
CLIP_B32 = Path(os.environ.get('RAREVLM_CLIP_B32', ROOT/'artifacts/models/clip-vit-base-patch32'/B32_REVISION))
CLIP_B16 = Path(os.environ.get('RAREVLM_CLIP_B16', ROOT/'artifacts/models/clip-vit-base-patch16'/B16_REVISION))
CIFAR100_ROOT = Path(os.environ.get('RAREVLM_CIFAR100', ROOT/'data/cifar100'))
INAT2018_ROOT = Path(os.environ.get('RAREVLM_INAT2018', ROOT/'data/inat2018'))
