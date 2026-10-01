# Completed frozen results

Generated from [main_results.csv](../results/main_results.csv). Accuracy, Macro-F1 and ECE are percent; NLL is unscaled. SD is descriptive training-seed variation; seeds share final images. ZS is one deterministic evaluation per setting.

## imagenet_lt_b32

| Method | OA | Many | Medium | Few | Macro-F1 | NLL | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZS | 59.42 | 61.03 | 58.17 | 59.22 | 58.21 | 1.684 | 1.40 |
| CE | 61.38 | 76.10 | 56.25 | 37.76 | 60.10 | 1.495 | 8.45 |
| LA | 65.97 | 70.62 | 64.22 | 59.01 | 65.40 | 1.290 | 3.39 |
| Fixed CE, empirical α=1 | 65.35 | 68.91 | 63.14 | 63.03 | 64.92 | 1.321 | 3.26 |
| Fixed CE, val-α | 65.36 | 69.87 | 62.90 | 61.23 | 64.86 | 1.321 | 3.40 |
| Fixed CE, train-only P2P-style | 66.19 | 70.38 | 64.54 | 60.16 | 65.74 | 1.283 | 2.63 |
| LA, train-only P2P-style | 66.66 | 70.70 | 65.00 | 61.10 | 66.23 | 1.263 | 2.57 |

Validation-selected α by seed: `0:0.75;42:1;200:1`.

## imagenet_lt_b16

| Method | OA | Many | Medium | Few | Macro-F1 | NLL | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZS | 63.99 | 65.17 | 63.21 | 63.35 | 62.83 | 1.475 | 1.64 |
| CE | 67.20 | 80.34 | 63.17 | 44.23 | 66.06 | 1.235 | 7.00 |
| LA | 71.04 | 74.96 | 69.97 | 63.73 | 70.56 | 1.071 | 3.14 |
| Fixed CE, empirical α=1 | 70.81 | 73.62 | 69.40 | 67.83 | 70.42 | 1.084 | 2.57 |
| Fixed CE, val-α | 70.83 | 75.41 | 68.93 | 64.57 | 70.31 | 1.082 | 2.85 |
| Fixed CE, train-only P2P-style | 71.58 | 74.89 | 70.53 | 65.95 | 71.18 | 1.053 | 2.08 |
| LA, train-only P2P-style | 71.67 | 75.18 | 70.63 | 65.39 | 71.30 | 1.047 | 2.48 |

Validation-selected α by seed: `0:0.75;42:0.75;200:1`.

## imagenet_lt_b32_online

| Method | OA | Many | Medium | Few | Macro-F1 | NLL | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZS | 59.42 | 61.03 | 58.17 | 59.22 | 58.21 | 1.684 | 1.40 |
| CE | 61.54 | 76.99 | 56.42 | 35.81 | 60.05 | 1.471 | 6.03 |
| LA | 66.64 | 71.65 | 64.71 | 59.28 | 66.08 | 1.254 | 1.20 |
| Fixed CE, empirical α=1 | 66.11 | 69.68 | 64.14 | 62.90 | 65.58 | 1.281 | 0.84 |
| Fixed CE, val-α | 66.11 | 69.68 | 64.14 | 62.90 | 65.58 | 1.281 | 0.84 |
| Fixed CE, train-only P2P-style | 66.98 | 70.97 | 65.43 | 61.14 | 66.46 | 1.246 | 1.01 |
| LA, train-only P2P-style | 67.24 | 71.41 | 65.43 | 61.80 | 66.81 | 1.230 | 0.73 |

Validation-selected α by seed: `0:1;42:1;200:1`.

## inat2018_b32

| Method | OA | Many | Medium | Few | Macro-F1 | NLL | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZS | 3.48 | 5.18 | 3.26 | 3.32 | 1.98 | 7.560 | 2.50 |
| CE | 5.05 | 28.70 | 3.09 | 1.57 | 2.35 | 6.834 | 6.63 |
| LA | 5.69 | 18.74 | 4.67 | 3.70 | 3.44 | 6.592 | 1.72 |
| Fixed CE, empirical α=1 | 4.00 | 7.06 | 2.81 | 4.52 | 2.42 | 6.835 | 3.59 |
| Fixed CE, val-α | 5.35 | 26.01 | 3.47 | 2.48 | 2.71 | 6.740 | 3.68 |
| Fixed CE, train-only P2P-style | 5.83 | 23.31 | 4.57 | 3.05 | 3.64 | 6.609 | 0.55 |
| LA, train-only P2P-style | 6.01 | 19.93 | 5.07 | 3.75 | 3.99 | 6.486 | 0.36 |

Validation-selected α by seed: `0:0.25;42:0.25;200:0.25`.

## Interpretation and boundaries

ImageNet-LT B/32 CE−ZS gives Many +15.06 pp/Few −21.46 pp; val-calibrated fixed CE improves OA/Few by +3.98/+23.47 pp. Complete B/16 and online views preserve the direction on the same dataset. iNaturalist gives Many +23.52/Few −1.74 pp; val-α=.25 and P2P improve CE OA/Few, while empirical α=1 loses OA 1.05 pp.

iNaturalist uses 8,142 scientific-name prototypes, frozen encoders and a lightweight adapter. OA about 3.48–6.01% reflects limited absolute recognition ability, not SOTA or deployment performance. No controlled comparison identifies which single factor causes the low accuracy.

Fixed-checkpoint recovery is intervention sufficiency, not a causal percentage or evidence that representations never changed. Shared images and only three final iNaturalist images/class limit generalization. Untested ARS/gradient methods are not ranked or declared disproven. [Detailed analysis](ANALYSIS.md), [protocol](PROTOCOL.md), and [source provenance](../results/provenance.json).
