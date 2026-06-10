# Hierarchical Reconciliation Experiments

These notebooks explore M5 reconciliation with Nixtla `hierarchicalforecast`.
Run them on capped data first:

```bash
M5_N_SERIES=500 M5_LAST_N_DAYS=200 make prep
```

The default production path remains `configs/m5/hier.yaml` and `make cv-hier`.
The expanded method sweep uses `configs/m5/hier_experiments.yaml` through the
same recipe-driven code path as the CLI.

## Notebooks

| Notebook | Purpose |
|---|---|
| `00_method_map.ipynb` | Build the M5 12-level hierarchy and document which reconciliation methods are appropriate for grouped M5 data. |
| `01_capped_cv_experiments.ipynb` | Run capped rolling-origin CV over BU, MinTrace, and ERM variants, then score WRMSSE. |

## Method notes

- **BottomUp** preserves item-store forecasts and improves aggregate coherence by construction. It is a useful guardrail when bottom-level models are strongest.
- **TopDown forecast proportions** can stabilize sparse item-store series by borrowing the total forecast, but may lose item-level signal. _removed because Nixtla requires a strict hierarchical tree_
- **MinTrace OLS / WLS** adjusts all levels jointly. OLS is the cheap baseline; structural WLS accounts for aggregation size; variance WLS uses in-sample residual variance where available.
- **MinTrace shrinkage** uses residual covariance shrinkage and is usually the strongest principled reconciler when residual history is adequate, but it is heavier on full M5.
- **ERM** learns a reconciliation matrix from fitted values. It can help when unbiasedness assumptions behind MinTrace are weak; `reg_bu` regularizes toward BottomUp when data is limited.
- **TopDown** and **MiddleOut** are intentionally excluded. Nixtla supports them only for strictly hierarchical trees, while M5 is a grouped hierarchy with overlapping product and geography levels.
