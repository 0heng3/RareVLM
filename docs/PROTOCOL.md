# Controlled protocol

## Shared model and supervision

- CLIP image/text encoders are frozen and evaluated without gradients. Class names use one prompt: `a photo of a {}.`. Text prototypes and CLIP logit scale remain fixed.
- Adapter: zero-initialized residual `512→64→512`, LayerNorm/GELU; 66,240 parameters. This uses the earlier projects' bottleneck residual idea, with a different normalization/activation implementation.
- CE and LA use the same data, initialization seed, sampler and optimizer budget. LA is `CE(z + log π_train, y)` with τ=1; evaluation uses z.
- AdamW lr=.001, weight decay=.01, batch=256, 20 epochs, seeds `{0,42,200}`. Validation OA selects an epoch, earliest tie. Final labels never select parameters or routes.
- Many >100, Medium 20–100, Few <20, based on actual counts after validation holdout.

## Dataset-specific splits

| Dataset | Train | Selection validation | Final evaluation |
| --- | ---: | ---: | ---: |
| ImageNet-LT | 115,846 | 20,000 | 50,000 |
| iNaturalist 2018 | 429,371 | 8,142 | 24,426 |

iNaturalist official train has 437,513 images. Hold out one image per class using minimum SHA256(`2026|class_id|relative_path`), leaving actual train counts 1–999. Official val (3 images/class) is locked as final top-1 research evaluation; the code's `test` key refers to this official val, not competition test/top-3. Scientific species names must match the official un-obfuscated table. This version is a controlled configuration with limited absolute recognition performance.

CIFAR-100-LT is a separate synthetic long-tail pilot: hold out 50 official-train images/class, then sample an actual IR=100 with max/min 400/4. Three frequency-to-semantic-class permutations are not interchangeable with training seeds.

## Fixed-checkpoint controls

Empirical correction is `z' = z − α log π_train`. Report fixed α=1 and validation-selected α separately. Grid `{0,.25,.5,.75,1,1.5,2}`, maximum internal validation OA, smaller-α tie. No adapter or prototype update.

For train-only P2P-style, use float64 softmax at the original temperature and uniform target u:

```text
CE: q = mean_train softmax(z); z' = z + log u − log q
LA: qm = mean_train softmax(z + log π)
    qbar ∝ qm / π; z' = z + log u − log qbar
```

Strength is fixed at 1. These controls are not the complete Prior2Posterior paper experiment with additional holdout/tuning. `calibrate.py` saves selection before reading final features and binds source checkpoint and feature hashes. iNaturalist's runner already executes these controls.

## Online training views

ImageNet-LT B/32 online training uses RandomResizedCrop224, scale(.08,1), ratio(3/4,4/3), bicubic antialias, horizontal flip p=.5 and CLIP normalization. RNG is keyed by `(seed,epoch,sample_index)`. CE/LA share each augmented tensor and one frozen CLIP encoding; adapters and AdamW states remain independent. Physical image microbatch64 accumulates to effective batch256. Validation/final center-crop caches remain fixed, without TTA.

## Interpretation

Report OA, Many/Medium/Few, Macro-F1, NLL, ECE and complete seed results. Shared final images mean seed×image pairs are not independent images. Recovery of net accuracy under fixed representations is intervention evidence, not an attributable causal percentage. Dataset/class count/image domain/text-name changes are not a single-factor comparison. Downstream split separation does not prove absence of CLIP pretraining overlap.

Simple calibration improving accuracy does not establish that untested ARS/gradient coordination mechanisms are ineffective. The project ends with documented empirical boundaries and engineering reproduction.
