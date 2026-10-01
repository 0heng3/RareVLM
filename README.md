# RareVLM: Long-Tailed Vision-Language Model Adaptation

Independent CV/VLM research and engineering project studying supervised CLIP adaptation under long-tailed class distributions. Frozen CLIP image/text encoders, fixed text prototypes, and a residual adapter with **66,240 trainable parameters** provide controlled CE/LA comparisons and fixed-checkpoint calibration interventions.

The project connects questions from TS-MOF/AES/GALE to pretrained VLM adaptation. Its contribution is a reproducible evaluation framework and evidence about mechanism boundaries. LA and train-only P2P-style correction are existing mechanisms, not proposed new algorithms.

## Install and run in minutes

Python 3.10 or 3.11 is recommended. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smoke.py
```

The smoke run uses synthetic features on CPU, needs no image data or model downloads, and exercises CE/LA training, validation selection, checkpoint saving, and frozen-checkpoint corrections. **Its numbers are not research results.** A CPU-only installation can install PyTorch/torchvision from the [official CPU wheel index](https://download.pytorch.org/whl/cpu) before installing the remaining requirements.

## Run a real CIFAR-100-LT pilot

```bash
python prepare_assets.py --model b32 --cifar100
python extract_features.py --output artifacts/cifar100/features.pt --device cuda:0
python run_pilot.py --features artifacts/cifar100/features.pt \
  --output-dir artifacts/cifar100/runs --device cuda:0 \
  --epochs 20 --batch-size 256 --train-seeds 0 42 200 --methods ce la ts
python summarize.py --input-dir artifacts/cifar100/runs --output-dir artifacts/cifar100/summary
```

Feature extraction over the complete real dataset takes longer than the CPU smoke. The pilot freezes IR=100 after validation holdout and uses three class-frequency permutations. `ts` is the minimal TailSpec-static candidate tested only in this pilot.

## ImageNet-LT, B/16, online augmentation and iNaturalist

See [reproduction commands](docs/REPRODUCE.md) for each dataset and training condition, [protocol](docs/PROTOCOL.md) for selection and calibration rules, and [completed results](docs/RESULTS.md) for numerical evidence and limitations. ImageNet/iNaturalist images, model weights, feature banks and per-image experimental artifacts are supplied or generated separately.

Portable defaults are `artifacts/models/` and `data/`. Override with CLI flags or `RAREVLM_CLIP_B32`, `RAREVLM_CLIP_B16`, `RAREVLM_CIFAR100`, and `RAREVLM_INAT2018`. B/16 checkpoints retain the pinned revision directory name. Asset preparation verifies the released weight/category hashes; generated manifests bind labels, split indices, counts, and paths.

## Findings

- ImageNet-LT B/32: CE adaptation increased Many accuracy by **15.06 pp** and reduced Few by **21.46 pp**, averaged over three seeds. A fixed CE adapter with validation-selected prior correction improved OA/Few from **61.38/37.76%** to **65.36/61.23%**.
- The direction reproduced using a complete B/16 checkpoint and online crop/flip training on the same ImageNet-LT. CE/LA online runs consume exactly the same augmented views while CLIP remains frozen.
- iNaturalist 2018: CE Many/Few changes were **+23.52/−1.74 pp**. LA, validation-selected correction and train-only P2P-style improved CE OA/Few in all three seeds. Empirical **α=1 reduced OA by 1.05 pp**, so unit-strength correction is not universal.

The iNaturalist configuration uses 8,142 official scientific-name prototypes and achieves approximately 3.48–6.01% OA; its absolute recognition ability is limited. The experiments establish recovery through decision interventions, not a unique causal explanation or superiority over untested optimization methods.

## Layout

```text
methods.py                     residual adapter, fixed prototype classifier, CE/LA/TailSpec-static
smoke.py                       CPU synthetic execution check
prepare_assets.py               pinned public assets and official species-name table
extract_*.py, run_*.py          dataset/backbone-specific extraction and training
online_aug_dataset.py          sample/epoch/seed-bound random crop and flip
calibrate.py                   empirical / validation-selected / train-only P2P controls
summarize_runs.py               one-manifest summary, without mixing protocols
docs/                          commands, protocol, results and source mechanism context
SOURCE_ORIGINS.json             source hashes and portable-edition changes
```

The portable edition changes local path defaults and binds B/16 checks to the supplied validated manifest. Model weights, losses, adapter structure, calibration formulas and training budgets retain the research settings. Original raw experiments remain in their own artifacts; this repository contains the minimal runnable distribution.
