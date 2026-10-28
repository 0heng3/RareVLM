# RareVLM

[简体中文](README.md) · English

### Vision-Language Adaptation: Long-Tail Analysis & Generative Fine-Tuning

An independent CV/VLM research engineering project studying long-tail adaptation, controlled multimodal SFT, and the quality/cost effects of visual input budgets and numerical inference paths.

**2026-10-28 update — matched full-scene SFT/DPO:** replacing 128 word-crop replay slots with fixed ST-VQA scene supervision produced **88.027211%** on the complete internal 1,617-question evaluation, versus 87.198516% at the common start and 87.631416% for the historical strong DPO control. SFT gained **0.828695 pp**, conditional image-cluster 95% CI **[0.193496, 1.502531]**, over the start; its 0.395795 pp lead over the strong control remains inconclusive. The preregistered new CE+DPO candidate scored 87.303649%, **0.723562 pp below matched SFT**, and failed its gates. Both 32-update arms and all 5,154 new evaluation generations completed. This is single-seed, historically used internal evaluation, not stable superiority or external confirmation. The fixed training branch is closed; a separately registered frozen-Final source-holdout evaluation makes **zero parameter updates**. [Results and limitations](docs/VLM_STVQA_REPLAY_V1_RESULTS.md) · [Portable metrics](results/vlm_stvqa_replay_v1_metrics.json) · [Source-holdout protocol](docs/VLM_STVQA_HOLDOUT_TRANSFER_V1_PROTOCOL.md).

![Python 3.10 / 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)

The original CLIP long-tail study is frozen. The 2026-10-12 archived Qwen study records **17 training runs and 101 evaluation cells** across separately recorded protocols, including the three-seed SFT study, internal-dev diagnostics, a completed 12-cell BF16/1024 formal reevaluation, a two-condition weighted-SFT pilot and a four-condition region-allocation diagnostic. Evaluation cells across protocols are not independent replications. The weighted pilot did not pass its rare-answer advancement gate. Automatic text regions helped against two-image controls but remained **5.850 pp below the single full-image reference**. Both results are retained, with no automatic seed expansion or official evaluation. The earlier engineering pilot is archived separately. [Completed inference results](docs/VLM_OPTIMIZATION_RESULTS.md) · [Research and learning route](docs/JOB_ROUTE.md).

[Replay-coverage pilot results](docs/VLM_OBJECT_REPLAY_COVERAGE_V1_RESULTS.md) are `completed_negative`: wide32 improved internal object accuracy from 95.614% to 97.661% (+2.047 pp), while Text fell from 87.112% to 86.964% (−0.148 pp). Five of eight gates passed; Text noninferiority and rare/frequent point guards failed. The repeat8x4 control cannot replace the candidate. No seeds 0/200 or candidate external/POPE evaluations were added; the real-improvement goal remains active. [Portable metrics](results/vlm_object_replay_coverage_v1_metrics.json).

![Overview: frozen image and text encoders feed a fixed prototype classifier; only the 66,240-parameter visual residual adapter is trained. Fixed-CE calibration changes logits, not model parameters.](assets/overview.png)

## Why RareVLM

CLIP already provides a zero-shot reference. Does ordinary CE supervision preserve that balance when the downstream training set is long-tailed? RareVLM answers this with paired residual-adapter runs, fixed-checkpoint interventions, and checks across complete checkpoints, training views and two real long-tailed datasets.

## Key findings

1. **CE produces a head/tail tradeoff.** On ImageNet-LT B/32, Many accuracy increases **15.06 pp**, while Few decreases **21.46 pp** over three training seeds.
2. **A fixed CE representation can recover performance.** Validation-selected prior correction raises OA/Few from **61.38/37.76%** to **65.36/61.23%** without updating the adapter, encoders or text prototypes.
3. **The direction repeats under different settings.** Complete B/16 and online crop/flip training reproduce it on ImageNet-LT; iNaturalist 2018 also shows CE Many↑/Few↓ and improvements from some fixed-CE corrections.
4. **Correction strength is conditional.** iNaturalist selects **α=0.25 in 3/3 seeds**; fixed α=1 improves Few but loses **1.05 pp OA**. Unit-strength empirical correction is not universal.

![CE-minus-zero-shot accuracy changes: Many gains and Few losses across four completed conditions. Bars show three-seed means, dots show seeds, and whiskers show seed SD, not confidence intervals.](assets/main_results.png)

## Main results

Accuracy in %, mean of seeds `{0,42,200}`. ZS is a deterministic reference. Every row below uses **validation-selected CE calibration**; individual α values and all fixed-strength/P2P controls are in the [machine-readable table](results/main_results.csv).

| Setting | ZS OA / Few | CE OA / Few | LA OA / Few | Fixed CE + val-α OA / Few |
| --- | ---: | ---: | ---: | ---: |
| ImageNet-LT · B/32 · fixed view | 59.42 / 59.22 | 61.38 / 37.76 | 65.97 / 59.01 | 65.36 / 61.23 |
| ImageNet-LT · B/16 · fixed view | 63.99 / 63.35 | 67.20 / 44.23 | 71.04 / 63.73 | 70.83 / 64.57 |
| ImageNet-LT · B/32 · online views | 59.42 / 59.22 | 61.54 / 35.81 | 66.64 / 59.28 | 66.11 / 62.90 |
| iNaturalist 2018 · B/32 · fixed view | 3.48 / 3.32 | 5.05 / 1.57 | 5.69 / 3.70 | 5.35 / 2.48 |

iNaturalist uses 8,142 official scientific-name prototypes. Its approximately **3.48–6.01% OA** under this frozen-encoder/fixed-classifier setup reflects limited absolute recognition ability; these are controlled research results, not deployment or SOTA claims. Official val is locked for final top-1 research evaluation, with selection confined to held-out official-train images. [Full results and boundaries](docs/RESULTS.md).

## Early generative VLM engineering pilot

**Completed:** Qwen2.5-VL-3B-Instruct → image-disjoint TextVQA instructions → QLoRA SFT → paired answer generation → local inference verification. Four V100 GPUs trained **1,843,200 language LoRA parameters (0.049%)**, with a frozen vision/base model, NF4 language weights and FP32 compute. One epoch used 1,024 training images, 128 diagnostic-validation images and 64 updates/rank; peak training allocated memory was **5.67 GiB/device**.

| Same NF4 / FP32 configuration, 256 final images | Normalized VQA consensus % | Any-human-answer match % |
| --- | ---: | ---: |
| Pretrained instruct | 68.55 | 70.70 |
| QLoRA SFT, fixed final epoch | 71.52 | 75.39 |

Paired consensus change: **+2.97 pp**, with a conditional image-bootstrap interval **[-0.20, 6.25] pp**. This single-seed subset and compact normalization do not establish robust gains or full-benchmark performance. The result is separate from CLIP OA/Few metrics. [Executed results and limitations](docs/VLM_SFT_RESULTS.md) · [Metrics/environment](results/vlm_sft_metrics.json) · [Run and CLI verification](results/vlm_sft_verification.json) · [Reproduce](docs/VLM_SFT.md).

## Completed generative adaptation and performance optimization

The deeper study compares approximately 1.84M trainable LoRA parameters across q/v rank8, q/k/v/o rank4, and q/v rank4 plus visual merger. Stage1 uses 10k QA, three seeds per method; the frozen internal-dev selection chooses two methods for 32,985-QA stage2 training. All 15 training runs and 67 original-protocol evaluations completed and passed verification. No formal-val-based checkpoint selection was added.

| Full stage2, 256 visual-token cap, NF4/FP32 | TextVQA official val | DocVQA ANLS ×100 | MMStar subcategory macro |
|---|---:|---:|---:|
| Fixed NF4 base | 69.030 | 68.730 | 54.732 |
| q/v rank8, mean ± seed SD | 72.166 ± 0.062 | 69.901 ± 0.176 | 54.018 ± 0.314 |
| q/v rank4 + merger, mean ± seed SD | 72.389 ± 0.085 | 69.954 ± 0.066 | 54.941 ± 0.577 |

Two later matrices add **12 complete internal-dev cells, 19,404 predictions, and no training**. Each cell evaluates 1,617 QA / 1,024 images. The adapter is the predeclared stage2 q/v seed42 checkpoint. These dev scores stay separate from the official-val and three-seed table above.

| Fixed NF4/FP32 checkpoint, internal dev | Base score | SFT score | SFT generation QA/s |
|---|---:|---:|---:|
| 256 visual-token cap | 71.738 | 74.601 | 2.919 |
| 512 visual-token cap | 78.299 | 82.158 | 1.511 |
| 1024 visual-token cap | 83.562 | 86.506 | 0.686 |

The SFT 1024−256 difference is **+11.905 pp**, with a paired image-cluster exploratory interval **[10.085, 13.770] pp**, and approximately **4.26×** amortized generation time. The base also benefits, so this is input-information evidence rather than a newly established LoRA mechanism. Actual mean visual tokens are 241.607, 489.021, and 938.856; caps are not actual token counts.

![Fixed-checkpoint visual input quality and cost](assets/vlm_optimization_v1/budget_quality_cost.png)

| Fixed SFT checkpoint, cap256, internal dev | Score | Generation QA/s | Generation + preprocessing + H2D QA/s | Peak allocated GiB |
|---|---:|---:|---:|---:|
| NF4/FP32 | 74.601 | 2.918 | 2.692 | 5.346 |
| Original/FP32 | 74.904 | 3.285 | 2.996 | 14.317 |
| Original/BF16 | 75.343 | 8.015 | 6.510 | 7.207 |

Original-weight BF16 provides **2.44× generation throughput**, **2.17× three-phase throughput**, and **49.66% lower peak allocated memory** versus original-weight FP32 at batch4 on A100. This is fixed-checkpoint inference, not BF16 retraining or production serving. Quality difference is +0.439 pp with interval [-0.091, 0.957]; precision equivalence was not established. Three-phase timing excludes loading, preflight, file writes, and controller overhead. No same-GPU contention was observed in sampled monitoring.

![Fixed256 numerical-path quality and cost](assets/vlm_optimization_v1/precision_quality_cost.png)

These two matrices alone did not measure high-budget BF16. A subsequent [joint diagnostic](docs/VLM_JOINT_BF16_RESULTS.md) completed four BF16 cap512/cap1024 dev cells: the fixed SFT checkpoint at cap1024 scored **86.926**, with **2.172 generation QA/s** and **8.652 GiB** peak allocated. These are one actual configuration, while the earlier 2.44× ratio still belongs to the cap256 FP32/BF16 comparison.

The separate [formal BF16/1024 protocol](docs/VLM_FORMAL_BF16_V1_PROTOCOL.md) completed all **12 cells / 47,396 predictions**, evaluating the base once and three existing q/v checkpoints with no new training:

| Original BF16, cap1024; public benchmarks | Base | q/v checkpoints, mean ± sample SD | Mean difference, pp |
|---|---:|---:|---:|
| TextVQA, 5,000 QA | 79.560 | 81.138 ± 0.169 | +1.578 |
| DocVQA ANLS, 5,349 QA | 91.631 | 91.819 ± 0.019 | +0.187 |
| MMStar 18-subcategory macro, 1,500 QA | 56.289 | 55.869 ± 0.245 | −0.420 |

![Completed formal BF16/1024 reevaluation](assets/vlm_formal_bf16_v1/formal_quality.png)

Each TextVQA fixed-checkpoint paired image-cluster exploratory interval is above zero. DocVQA and MMStar intervals cross zero; this does not establish equivalence or general capability retention. Actual supervised-target rare/mid frequency slices do not improve in their point estimates and have no image-sampling intervals; answer frequency is only an exposure proxy. Public validation has prior exposure, pretraining overlap is unknown, and training provenance mixes V100/A100. Nine pointwise exploratory intervals are unadjusted for multiplicity. [Portable formal metrics](results/vlm_formal_bf16_v1_metrics.json) · [Formal figure provenance](assets/vlm_formal_bf16_v1/FIGURE_PROVENANCE.json) · [Portable two-matrix metrics](results/vlm_optimization_v1_metrics.json) · [Detailed evidence and code](docs/VLM_OPTIMIZATION_RESULTS.md) · [Frozen SFT protocol](docs/VLM_DEEP_V2_PROTOCOL.md).

## Frequency-weighted SFT and region allocation

The weighted-SFT pilot trained fresh uniform and frequency-weighted q/v rank8 adapters from matched initialization and QA order: **32,985 QA, 2,062 updates and 1,843,200 trainable parameters per run**. Both were evaluated using the same original-weight BF16/cap1024 path. The single-seed internal-dev score changed **87.112→86.877**; rare supervised-answer score changed **87.398→87.281**. The rare-answer gate failed. [Protocol, results and limits](docs/VLM_WEIGHTED_SFT_RESULTS.md) · [Machine-readable results](results/vlm_weighted_sft_v1_metrics.json).

The subsequent region diagnostic reused the uniform seed42 final adapter, with **0 new training runs and 6,468 predictions**. Each QA used exactly **1,024 merged visual tokens**. Whole-image and half-budget views preserve aspect ratio with gray letterboxing; a CPU text detector supplies geometry and confidence only, without OCR recognition text or reference answers. Random controls preserve the selected crop's width and height.

| New letterbox protocol, internal dev; 1,617 QA | Overall consensus, % | Rare supervised answers, % |
|---|---:|---:|
| One full-image view, 1,024 tokens | **84.886** | **82.310** |
| Two full-image views, 512 + 512 | 77.372 | 76.491 |
| Full image + predicted text region, 512 + 512 | 79.035 | 78.450 |
| Full image + random same-shape region, 512 + 512 | 76.964 | 76.491 |

Text regions outperform random regions by **2.072 pp** (exploratory image-cluster 95% interval **[0.752, 3.368]**) and duplicate full-image views by **1.664 pp [0.339, 3.020]**. They trail the single full-image reference by **5.850 pp [−7.589, −4.133]**. The predeclared gate checks the two-image controls, so `gate=true` does not establish the best overall method. Rare-answer gains over the controls have intervals crossing zero; long-tail improvement and three-seed stability are not established.

Allocated memory is approximately **8.420 GiB** for all four conditions. Cached-view VLM phase throughput is **1.74–1.79 QA/s**; CPU text detection adds **554.52 seconds for 1,024 unique images**. These timings do not establish end-to-end acceleration. Equal visual tokens do not imply equal information, total prompt length, FLOPs or runtime. This new letterbox protocol is separate from earlier native-resize scores. [Full evidence and code](docs/VLM_REGION_TOKEN_RESULTS.md) · [Machine-readable results](results/vlm_region_token_v1_metrics.json).

## Quick start

Python 3.10/3.11; run from repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
pip install -e . --no-deps
python -m smoke
```

This is a **CPU-only synthetic execution check**, with no image/model download. It exercises actual CE/LA training, validation selection, checkpoint saving and fixed-checkpoint controls. Its numbers are not research evidence. For a CPU-only PyTorch installation, install torch/torchvision from the [official CPU wheel index](https://download.pytorch.org/whl/cpu) before the remaining requirements.

```bash
pip install -r requirements/dev.txt
python -m pytest -q
python tools/check_release.py
```

The project uses local tests, CPU smoke and release verification. GitHub Actions is disabled.

Local verification passed **45 CPU contract tests**, including generated-VQA masking/split/metric contracts and a two-rank Gloo gradient check against a centralized weighted objective. The synthetic CPU smoke also passed. These checks validate software contracts; their synthetic outputs are not research results.

## Experimental design and reproduction

The generative study uses frozen Qwen vision/LLM weights, explicit language/merger LoRA targets, image-disjoint TextVQA data, per-question assistant-only loss, resumable DDP training and task-specific generated-answer metrics. The historical NF4/FP32 path remains frozen for V100 compatibility; a separate inference bridge loads original weights and LoRA in BF16 on A100. Install `requirements/vlm.txt`; follow [early pilot](docs/VLM_SFT.md), [deep SFT](docs/VLM_DEEP_V2_PROTOCOL.md), [visual budget](docs/VLM_BUDGET_PROBE_V1_PROTOCOL.md), and [precision bridge](docs/VLM_PRECISION_BRIDGE_V1_PROTOCOL.md). New GPU jobs wait for idle devices and preserve existing processes. These protocols remain independent of CLIP OA/Few.

- Frozen CLIP image/text encoders; single `a photo of a {}.` prompt; normalized, fixed text prototypes and original logit scale.
- Zero-initialized residual **512→64→512** LN/GELU adapter: **66,240 trainable parameters**, exact identity at initialization.
- Paired CE/LA initialization, sample order, budget and selection. LA uses `CE(z + log π_train, y)`, τ=1; final evaluation uses raw z.
- Online training binds crop/flip RNG to `(seed, epoch, sample_index)` and shares one frozen CLIP image encoding between independent adapters/AdamW states. Final views have no augmentation or TTA.

[Protocol](docs/PROTOCOL.md) · [ImageNet-LT / B/16 / online / iNaturalist commands](docs/REPRODUCE.md) · [Four frozen configuration snapshots](configs/imagenet_lt_b32.json).

The CLIP JSON configurations document settings; existing runners use CLI commands in `REPRODUCE.md`. CLIP paths default to project-local `data/` and `artifacts/`, with CLI/environment overrides. Dataset images and model weights are acquired separately; [asset preparation](src/prepare_assets.py) checks pinned weights and the official iNaturalist species table.

The newer GPU configurations and launchers are frozen execution snapshots of the original persistent workspace. GPU replay requires separately prepared model/data/adapter artifacts, a compatible numerical environment and explicit runtime path binding. Snapshot defaults under `/opt/data/private` are not a promise that a clean clone can replay these experiments unchanged. Raw training checkpoints, full prediction caches, detector environments and image-bearing learning pages remain local; the repository includes source, exported metrics, provenance hashes and learning-page generators.

## Analysis

[docs/ANALYSIS.md](docs/ANALYSIS.md) contains deeper evidence: ZS→CE error transitions, effective versus empirical priors, margins against fixed ZS competitors, representation drift/alignment, and the limits of causal interpretation. Source mechanisms and execution status are in [MECHANISMS.md](docs/MECHANISMS.md).

## Reproducibility & auditability

- [main_results.csv](results/main_results.csv): 28 final setting/method summaries; [seed_results.csv](results/seed_results.csv): 76 setting/seed records, including the reused ZS baseline.
- [provenance.json](results/provenance.json): model, manifest, ordered split and source-result hashes; prompt, seed, protocol and selection identity. Tables are generated by [export_results.py](tools/export_results.py) from completed JSON records.
- [Figure provenance](assets/figure_provenance.json): source CSV hash, transformations, mean/SD definition, individual seed values and exported image hashes.
- [Research code lock](results/research_code_lock.json): original runnable research modules stay frozen. [SOURCE_ORIGINS](metadata/SOURCE_ORIGINS.json) records source/portable hashes and path-related changes.
- [PACKAGE_MANIFEST](metadata/PACKAGE_MANIFEST.json): SHA-256 for release files. [check_release.py](tools/check_release.py) checks tables, images, links and file hashes.
- [tests/](tests): exact identity/parameter budget, label-free classifier API, frozen prototypes, LA semantics, split separation, immutable calibration inputs, and online view pairing across RNG states/workers.

## Research context

Questions from **TS-MOF / AES / GALE** motivated the study. The final VLM main matrix focused on CE/LA and existing fixed-checkpoint corrections. TailSpec-static was tested only in the CIFAR pilot; ARS, E-PCG, RD-PCGrad, GES and multi-expert fusion were not VLM main-matrix comparisons. Context includes [Logit Adjustment](https://openreview.net/forum?id=37nvvqkCo5), [LIFT](https://proceedings.mlr.press/v235/shi24g.html), and [Prior2Posterior](https://openaccess.thecvf.com/content/WACV2025/html/Bhat_Prior2Posterior_Model_Prior_Correction_for_Long-Tailed_Learning_WACV_2025_paper.html). No new calibration algorithm is claimed.

## Scope & limitations

Fixed-representation recovery establishes that decision interventions can help; it does not identify a unique causal source or prove representations are undamaged. B/32/B/16 change the complete CLIP checkpoint. Dataset, class count, domain and name system also vary. Shared final images make seed×image pairs dependent, and iNaturalist has only three final images per class. Downstream split separation cannot prove absence of CLIP pretraining overlap. Untested optimization candidates are not ranked or declared ineffective.

## Repository layout

```text
RareVLM/
├── src/             CLIP modules and the nine vlm_* research packages
├── tools/           analysis, learning readers, export and release checks
├── tests/           CPU contracts
├── configs/         original experimental configuration snapshots
├── requirements/    base, development and GPU dependency lists
├── results/         protocol-specific metrics and historical research locks
├── assets/          figures and figure provenance
├── docs/            protocols, results, reproduction and learning routes
├── metadata/        source origins, migration mapping and release manifest
├── pyproject.toml   editable installation, module discovery and pytest settings
├── README.md        Chinese overview
├── README.en.md     English overview
├── CITATION.cff
└── .gitignore
```

The `src/` layout retains existing module names: `import methods` and `python -m vlm_region.pipeline` still work after editable installation. Root-level script commands now use `python -m smoke` or `python -m run_imagenet_lt`. Source and project directories are resolved separately.

The layout migration records old/current paths and hashes without overwriting original configs, CSV/JSON evidence or research locks. Earlier runs bind the sources at commit `2049c0e`; exact historical recovery uses that commit in an independent checkout. Source changes in this version invalidate older run locks, so new runs use fresh output directories. [Architecture and migration](docs/ARCHITECTURE.md) · [Migration provenance](metadata/ARCHITECTURE_MIGRATION.json).

## License & citation

License publication is pending the maintainer's confirmation. Software citation metadata is provided in [CITATION.cff](CITATION.cff). Third-party datasets, model weights and source-project archives are obtained separately.
