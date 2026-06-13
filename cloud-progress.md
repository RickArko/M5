# Cloud Training Progress Log

## VM Configuration
- **Server Type**: ccx33 (8 vCPU / 32 GB RAM) — current run
- **Future**: ccx83 (24 vCPU / 96 GB RAM) for full M5 + hierarchy
- **Cost**: ~€0.06/hour (ccx33), ~€0.18/hour (ccx83)
- **Location**: fsn1 (Falkenstein, Germany)
- **Artifact Bucket**: s3://m5-terraform-training-vm/lgbm

## Current Run Progress

### Phase 1: Setup (COMPLETE)
- [x] Cloud-init bootstrap
- [x] Install uv + sync deps
- [x] Install awscli v2
- [x] Clone repo from main

### Phase 2: Data Download (COMPLETE)
- [x] Download M5 raw CSVs
- [x] Calendar, prices, sales data

### Phase 3: Prep (COMPLETE ✅)
- [x] Build long-format parquet
- [x] **46,796,220 rows** (30,490 series × 1,941 days)
- [x] Memory: 1,565.4 MB
- [x] Duration: ~30 seconds
- [x] Output: `data/processed/long.parquet`

### Phase 4: Cross-Validation (IN PROGRESS)
- [ ] CV Stats (Theta + AutoETS + SeasonalNaive)
  - Started: 05:34:09 UTC
  - Status: Running (36+ minutes)
  - ETA: ~10-15 more minutes

- [ ] CV LGBM (LightGBM global model)
  - Status: Pending
  - ETA: ~15 minutes

- [ ] CV Hier (Hierarchical Theta + BU/MinT)
  - Status: Pending
  - ETA: ~10 minutes

### Phase 5: Scoring (PENDING)
- [ ] Compute WRMSSE for all models
- [ ] Generate reports
- ETA: ~2 minutes

### Phase 6: Final Training (PENDING)
- [ ] Train on full data
- [ ] Push model artifact to S3
- ETA: ~10 minutes

### Phase 7: Cleanup (PENDING)
- [ ] Push CV artifacts to S3
- [ ] Push reports to S3
- [ ] Push processed data to S3 (if push_processed=true)
- [ ] VM poweroff

## Monitoring Log

### 2026-06-13 06:10 UTC
- Status: CV Stats running (36 minutes elapsed)
- CPU: 100% (single-core bounded)
- RAM: 4.8 GB (15% of 32 GB)
- Log size: 5.1 KB, 89 lines
- Artifacts dir: empty (CV hasn't written yet)

## Next Check
- Scheduled: 2026-06-13 06:20 UTC

## Post-Completion Checklist
- [ ] VM powered off automatically
- [ ] Artifacts pulled from S3
- [ ] Vue dashboard exported from CV artifacts
- [ ] Dashboard serves at http://localhost:5173
