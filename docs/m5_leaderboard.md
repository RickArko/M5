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
global model with ~6 features and 3 lags). Realistic expectations:

| Model in this repo               | Typical bottom-level WRMSSE | Rough leaderboard equivalent |
|----------------------------------|----------------------------:|------------------------------|
| SeasonalNaive(7)                 | ~0.95 – 1.10                | well below benchmarks        |
| Theta                            | ~0.80 – 0.90                | ≈ M5 Theta benchmark         |
| AutoETS('ZNA')                   | ~0.78 – 0.88                | ≈ M5 ETS benchmark           |
| LightGBM (Tweedie, 3 lags)       | ~0.65 – 0.72                | ≈ top-15% (silver/bronze edge) |
| Hierarchical Theta + MinTrace    | ~0.70 – 0.78                | ≈ top-25% on aggregate metric  |

To approach top-tier (≤0.55), you need: more features (price-bin lags, day-of-month
encodings, holiday-distance), per-store LightGBM models, weighted blends across
horizons, and post-hoc reconciliation. The repo's tenet is "fewer features, not
more" — so the LightGBM number above is the realistic ceiling without that work.

## Sources

- [M5 paper (Makridakis, Spiliotis, Assimakopoulos 2022 — IJF)](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [Mcompetitions/M5-methods (top submissions)](https://github.com/Mcompetitions/M5-methods)
- [Christophe Nicault — M5 Accuracy writeup](https://www.christophenicault.com/post/m5_forecasting_accuracy/)
- [Kaggle: M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy)
