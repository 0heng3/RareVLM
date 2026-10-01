# Reproduction commands

Run from repository root, after `pip install -r requirements.txt`. Full experiments require separately supplied images and a CUDA device; `python smoke.py` is the fast CPU execution check. Paths below are examples, not embedded machine paths. Use separate output directories for datasets, checkpoints and training conditions.

## ImageNet-LT B/32

Obtain ImageNet images through your existing dataset access and ImageNet-LT lists from the [OLTR release](https://github.com/zhmiao/OpenLongTailRecognition-OLTR). The image root contains `train/<wnid>/...` and `val/<wnid>/...`; the split directory contains `ImageNet_LT_train.txt`, `ImageNet_LT_val.txt`, `ImageNet_LT_test.txt`.

```bash
export RAREVLM_IMAGENET_ROOT=/path/to/ImageNet/CLS-LOC
export RAREVLM_IMAGENET_SPLITS=/path/to/ImageNet_LT
python prepare_assets.py --model b32
python imagenet_lt_manifest.py --split-dir "$RAREVLM_IMAGENET_SPLITS" \
  --image-root "$RAREVLM_IMAGENET_ROOT" --output artifacts/imagenet_lt/manifest.json
python extract_imagenet_lt_features.py --manifest artifacts/imagenet_lt/manifest.json \
  --image-root "$RAREVLM_IMAGENET_ROOT" --output-dir artifacts/imagenet_lt/features \
  --device cuda:0 --batch-size 128 --workers 6
python run_imagenet_lt.py --manifest artifacts/imagenet_lt/manifest.json \
  --feature-dir artifacts/imagenet_lt/features --output-dir artifacts/imagenet_lt/main \
  --device cuda:0 --train-seeds 0 42 200 --methods ce la
python posthoc_imagenet_lt.py --manifest artifacts/imagenet_lt/manifest.json \
  --feature-dir artifacts/imagenet_lt/features --main-results artifacts/imagenet_lt/main \
  --output-dir artifacts/imagenet_lt/posthoc --device cuda:0
python calibrate.py --manifest artifacts/imagenet_lt/manifest.json \
  --feature-dir artifacts/imagenet_lt/features \
  --source-result artifacts/imagenet_lt/main/imagenet_lt_seed0_ce.json \
  --output-dir artifacts/imagenet_lt/controls --device cuda:0
python summarize_runs.py --input-dir artifacts/imagenet_lt/main \
  --controls artifacts/imagenet_lt/controls --output artifacts/imagenet_lt/summary.json
```

Run `calibrate.py` for seed42/200 and LA as needed. CE outputs fixed α=1, val-selected α and P2P-style; LA outputs train-only Eq.29-style correction. `posthoc_imagenet_lt.py` reproduces the original three-seed B/32 val grid. All calibration uses the frozen checkpoint and original center-crop train features to estimate effective prior.

## Full B/16 checkpoint

Reuse only the B/32 data manifest. Re-encode all image/text features with the pinned complete B/16 model:

```bash
python prepare_assets.py --model b16
python extract_imagenet_lt_features_b16.py --manifest artifacts/imagenet_lt/manifest.json \
  --image-root "$RAREVLM_IMAGENET_ROOT" --output-dir artifacts/imagenet_lt_b16/features \
  --model-snapshot artifacts/models/clip-vit-base-patch16/57c216476eefef5ab752ec549e440a49ae4ae5f3 \
  --model-revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 --device cuda:0
python run_imagenet_lt_b16.py --manifest artifacts/imagenet_lt/manifest.json \
  --feature-dir artifacts/imagenet_lt_b16/features --output-dir artifacts/imagenet_lt_b16/main \
  --expected-weight-sha ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0 \
  --device cuda:0
python calibrate.py --manifest artifacts/imagenet_lt/manifest.json \
  --feature-dir artifacts/imagenet_lt_b16/features \
  --source-result artifacts/imagenet_lt_b16/main/imagenet_lt_b16_seed0_ce.json \
  --output-dir artifacts/imagenet_lt_b16/controls --device cuda:0
```

Repeat calibration for the other seeds; `summarize_runs.py` summarizes the resulting B/16 directory independently. The portable B/16 runner validates the supplied manifest and model fingerprints; machine-dependent original manifest hashes are not imposed.

## B/32 online augmentation

Use the original B/32 data/prototypes/center-crop evaluation cache. Train features are encoded from augmented raw images, not from the cached training feature bank.

```bash
python run_imagenet_lt_online_aug.py --manifest artifacts/imagenet_lt/manifest.json \
  --image-root "$RAREVLM_IMAGENET_ROOT" --feature-dir artifacts/imagenet_lt/features \
  --output-dir artifacts/imagenet_lt_online --seed 0 --device cuda:0 --workers 6
python calibrate.py --manifest artifacts/imagenet_lt/manifest.json \
  --feature-dir artifacts/imagenet_lt/features \
  --source-result artifacts/imagenet_lt_online/imagenet_lt_seed0_aug_ce.json \
  --output-dir artifacts/imagenet_lt_online/controls --device cuda:0
```

Repeat the paired online runner for seeds42/200. It saves CE/LA initialization, order and crop/flip audit information. Validation/final evaluation use fixed center crops, no TTA.

## iNaturalist 2018

Use the [official dataset release](https://github.com/visipedia/inat_comp/blob/master/2018/README.md). Put `train2018.json`, `val2018.json` and the `train_val2018/` image directory under a common root. Asset preparation downloads only the small official un-obfuscated category table, not the image archive.

```bash
export RAREVLM_INAT2018=/path/to/inat2018
python prepare_assets.py --model b32 --inat-categories
python inat2018_manifest.py --image-root "$RAREVLM_INAT2018" \
  --categories "$RAREVLM_INAT2018/categories.json" --output artifacts/inat2018/manifest.json
python extract_inat2018_features.py --manifest artifacts/inat2018/manifest.json \
  --output-dir artifacts/inat2018/features --device cuda:0 --batch-size 128 --workers 16
python run_inat2018.py --manifest artifacts/inat2018/manifest.json \
  --feature-dir artifacts/inat2018/features --output-dir artifacts/inat2018/main --device cuda:0
```

The runner includes three-seed CE/LA and fixed CE controls; full results are written to `main/summary.json`. Official val is final evaluation, with checkpoint/α selection confined to the held-out training images. The original model's scientific-name single-template accuracy is limited; retain negative results such as α=1 losing OA.
