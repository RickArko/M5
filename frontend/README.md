# M5 Accuracy Observatory

Vue dashboard for comparing M5 forecast accuracy across models, hierarchy levels, segments, and horizon days.

## Setup

From the repo root:

```bash
cd frontend
npm install
npm run export:data
npm run dev
```

Open the Vite URL printed by `npm run dev`. If `public/data/accuracy-dashboard.json` is absent, the app falls back to `public/data/accuracy-dashboard.sample.json`.

## Data Refresh

Generate CV artifacts first from the repo root, capped on local hardware:

```bash
M5_N_SERIES=500 M5_LAST_N_DAYS=200 M5_N_WINDOWS=1 make prep cv-stats cv-lgbm cv-hier
```

Then export dashboard data:

```bash
cd frontend
npm run export:data
```

The exporter reads:

```text
../data/processed/long.parquet
../artifacts/cv_*.parquet
```

and writes:

```text
frontend/public/data/accuracy-dashboard.json
```

That generated JSON is gitignored because it is derived from local artifacts.

## Test

```bash
cd frontend
npm test
```

## Production Build

```bash
cd frontend
npm run export:data
npm run build
npm run preview
```

Deploy the static `frontend/dist/` directory to any static host. For GitHub Pages, Netlify, Cloudflare Pages, S3, or nginx, build after `npm run export:data` so `dist/data/accuracy-dashboard.json` is included.
