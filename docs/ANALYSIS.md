# Mechanism analysis of the frozen study

These analyses use already completed checkpoints and predictions. P0/P0.5 were exploratory controls following the initial B/32 result; they are not retroactive preregistration. The compact release publishes final summaries and source hashes, not large feature/prediction archives.

## Same-image error transitions

Define a lost case by `ZS correct → CE wrong`, then compare LA and fixed-CE post-hoc recovery on that same image. Over three B/32 ImageNet-LT seeds, CE loses 15,336 seed×image pairs that ZS answered correctly. Few losses are **4,992/12,081 = 41.3%** of ZS-correct Few pairs; Medium/Many losses are 20.2%/5.4%.

Of the 4,992 lost Few pairs, LA recovers **3,298 (66.1%)**, while val-selected fixed-CE correction recovers **3,308 (66.3%)**. These are conditional recovery counts, not independent images or a causal percentage of degradation. Calibration also creates new errors; full-set OA/Few, not only the rescued subset, determines the net intervention result.

LA's original B/32 OA advantage over val-posthoc is about 0.62 pp. Full-set comparisons combine Many/Medium gains with fewer Few correct cases; this residual is not proof of representation preservation. B/16's corresponding mean OA residual is about 0.21 pp with one reversed seed. Later frozen residual checks did not support upgrading the project to a new-method route.

## Empirical and effective priors

The empirical prior is actual training class frequency. CE's effective prior is the mean train softmax prediction at the original model temperature. Fixed-strength P2P-style targets the uniform evaluation prior; it is not the paper's complete holdout/tuning procedure.

| Condition | Spearman(q_CE, π_train) | TV(q_CE, π_train) |
| --- | ---: | ---: |
| ImageNet-LT B/32 fixed view | 0.977–0.979 | 0.052–0.057 |
| ImageNet-LT B/16 fixed view | 0.980–0.982 | 0.046–0.052 |
| ImageNet-LT B/32 online | 0.977–0.983 | 0.049–0.054 |
| iNaturalist B/32 | 0.402–0.415 | 0.247–0.249 |

B/32 fixed-strength CE P2P versus empirical α=1 improves OA about 0.85 pp while reducing Few 2.87 pp. Effective prior is not a uniformly better metric choice. On iNaturalist, empirical α=1 loses OA, while val-α=.25 and P2P improve CE OA/Few. This supports distinguishing empirical frequency from model-dependent prediction marginals. The marginals also include semantic confusion and model capability; their discrepancy does not identify one causal factor.

## Margins with ZS-fixed competitors

For each final image, fix the top five non-target competitors using **ZS logits only**. Reuse those candidates for every CE seed; do not select competitors based on CE errors. Analyze

```text
Δr(x,k) = [z_CE,k − z_CE,y] − [z_ZS,k − z_ZS,y]
g(x,k) = ln n_k − ln n_y
```

After removing each image's common mean, B/32 fixed-view slopes are +0.574/+0.582/+0.567 logits per log-frequency unit. B/16 slopes are +0.543/+0.547/+0.575; B/32 online slopes are +0.603/+0.579/+0.584. Controlling ZS competitor strength preserves the direction. These are frequency associations, not causal identification; semantic/category difficulty remains mixed with frequency.

Each image contributes five candidates and three seeds. The 750,000 seed×image×competitor rows are not independent observations. Exploratory resampling used shared original image IDs; the webpage chart reports seed SD, not these conditional bootstrap intervals.

## Representation diagnostics

Report adapted-to-frozen cosine drift, true-class image/text alignment and classification margin. B/32 CE shows greater drift and lower Few alignment than ZS; LA differs in both representation and decisions. Holding CE fixed while changing logits leaves representation quantities unchanged but can improve margins and accuracy.

That intervention demonstrates recovery without updating the current representation. It cannot establish that representation never changed or that all lost accuracy came from a pure decision bias. Labels used for offline alignment/margins are evaluation inputs, not classifier inference inputs.

## Interpretation and source access

The completed evidence supports a reproducible head/tail tradeoff and conditional recovery through fixed-checkpoint interventions. Unit-strength empirical correction is dataset dependent. It does not justify a new adaptive-calibration claim, a ranking of untested gradient solvers, or a general rare-concept recognition claim.

The original analysis archives remain separate from the compact repository. Final per-seed metrics and source-result hashes are in [seed_results.csv](../results/seed_results.csv) and [provenance.json](../results/provenance.json); formulas and selection permissions are in [PROTOCOL.md](PROTOCOL.md).
