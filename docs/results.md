# M5 Forecasting Accuracy — public leaderboard reference

Quick reference for comparing our run's WRMSSE to the M5 Kaggle competition.

WRMSSE is computed across all 12 hierarchical aggregation levels in the
official M5 metric. **Our `m5.evaluation.wrmsse` scores the bottom level
(item × store) only** — see `CLAUDE.md` § Caveats. Hierarchical scoring
runs through `m5.cv.hier_cv` + `m5.hierarchy`. Bottom-level WRMSSE is
typically **lower** than the official 12-level WRMSSE because the bottom
level is the noisiest and dominates the aggregate.

## Final private leaderboard (5,558 teams, 7,092 participants)

| Rank   | Team / Method                    | Private WRMSSE (12-level) |
|-------:|----------------------------------|--------------------------:|
| 1      | YJ_STU (LightGBM, blended)       | ~0.520                    |
| 2      | Matthias Anderer (LightGBM)      | ~0.527                    |
| 3–5    | top-5 cluster                    | 0.52 – 0.55               |
| 50     | top-1% cutoff                    | ~0.55                     |
| 100    | top-2% cutoff                    | ~0.58                     |
| 190    | top-4% (silver-medal cutoff)     | 0.62408¹                  |
| 385    | top-7% (bronze-medal cutoff)     | 0.66695¹                  |

¹ Anchors from Christophe Nicault's writeup of his own submission.

## Organizer benchmarks (from M5 paper, Makridakis et al. 2022)

| Benchmark            | Description                              | Approx. private WRMSSE |
|----------------------|------------------------------------------|-----------------------:|
| Naive                | Last-value forecast                      | ~1.5                   |
| sNaive (s=7)         | Same-day-last-week (our included baseline) | ~1.0                  |
| SES                  | Simple exponential smoothing             | ~1.0                   |
| ETS                  | Auto-ETS                                 | ~0.88                  |
| Theta                | Theta method                             | ~0.87                  |
| ESX                  | ETS with exogenous (best M5 benchmark)   | ~0.671                 |
| **Winner vs top benchmark** | YJ_STU vs ESX                     | **22.4% improvement**  |

## What to expect from this repo

This is a **deliberately minimal** baseline (3 stats models + a single LightGBM
global model with ~6 features and 3 lags).

### Actual results — GCP `n2-highmem-8` run, 2026-05-10 (`20260510T052927Z`)

Full M5 (30,490 series × 400-day trailing window, h=28, n_windows=3, seed=42).
Code at git `1adcf65` (origin/ai).

| Model         | Bottom-only WRMSSE | **Official 12-level WRMSSE** | Kaggle equivalent |
|---------------|-------------------:|-----------------------------:|-------------------|
| **LGBM**      | 0.8402             | **0.6418**                   | **~rank 250–300, top 5%, silver-bronze edge** |
| AutoETS       | 0.8631             | 0.6724                       | ~rank 400, top 7%, bronze cutoff |
| Theta         | 0.8695             | 0.7376                       | ≈ M5 Theta benchmark |
| SeasonalNaive | 1.1115             | 0.8326                       | ≈ M5 sNaive benchmark |

LGBM wins **11 of 12** M5 levels; AutoETS wins only the Total level (by 0.002).
Most stable across CV folds: AutoETS (std = 0.0245); FVA vs sNaive (MAE):
LGBM **+19.1%**, AutoETS +15.5%, Theta +14.9%.

### To approach top-tier (≤0.55) you need

The repo's tenet is "fewer features, not more" — so the LGBM 0.642 number above
is roughly the ceiling without:

- More features (price-bin lags, day-of-month encodings, holiday-distance, weather)
- Per-store LightGBM models (10× the parameters, but matches the structural
  heterogeneity that's been lossy at the bottom level — see WI_2 0.93 vs WI_1 0.79)
- Weighted blends across horizons + recursive vs direct multi-step
- Post-hoc reconciliation (BU + MinTrace on top of LGBM, not just Theta)
- Tweedie variance-power tuning per category (FOODS_3 needs higher than HOBBIES)

## Sources

- [M5 paper (Makridakis, Spiliotis, Assimakopoulos 2022 — IJF)](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [Mcompetitions/M5-methods (top submissions)](https://github.com/Mcompetitions/M5-methods)
- [Christophe Nicault — M5 Accuracy writeup](https://www.christophenicault.com/post/m5_forecasting_accuracy/)
- [Kaggle: M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy)
