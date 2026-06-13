export const METRICS = [
  { key: 'wrmsse', label: 'WRMSSE', lowerIsBetter: true },
  { key: 'mae', label: 'MAE', lowerIsBetter: true },
  { key: 'rmse', label: 'RMSE', lowerIsBetter: true },
  { key: 'smape', label: 'sMAPE', lowerIsBetter: true, percent: true },
  { key: 'wmape', label: 'wMAPE', lowerIsBetter: true, percent: true },
  { key: 'bias', label: 'Bias', lowerIsBetter: false },
  { key: 'bias_pct', label: 'Bias %', lowerIsBetter: false, percent: true },
  { key: 'mase', label: 'MASE', lowerIsBetter: true },
]

export function metricMeta(metric) {
  return METRICS.find((m) => m.key === metric) ?? { key: metric, label: metric, lowerIsBetter: true }
}

export function rankRows(rows, metric = 'wrmsse') {
  const meta = metricMeta(metric)
  const copy = [...rows].filter((row) => Number.isFinite(Number(row[metric])))
  copy.sort((a, b) => {
    const diff = Number(a[metric]) - Number(b[metric])
    return meta.lowerIsBetter ? diff : Math.abs(Number(a[metric])) - Math.abs(Number(b[metric]))
  })
  return copy.map((row, index) => ({ ...row, rank: index + 1 }))
}

export function bestRow(rows, metric = 'wrmsse') {
  return rankRows(rows, metric)[0] ?? null
}

export function filterRows(rows, filters) {
  return rows.filter((row) => {
    if (filters.model && row.model !== filters.model) return false
    if (filters.level && row.level !== filters.level) return false
    if (filters.segment_axis && row.segment_axis !== filters.segment_axis) return false
    return !(filters.segment && row.segment !== filters.segment)
  })
}

export function availableValues(rows, key) {
  return [...new Set(rows.map((row) => row[key]).filter(Boolean))].sort((a, b) =>
    String(a).localeCompare(String(b)),
  )
}

export async function loadDashboardData() {
  const resp = await fetch('/data/accuracy-dashboard.json')
  if (resp.ok && resp.headers.get('content-type')?.startsWith('application/json')) {
    return resp.json()
  }
  const sample = await fetch('/data/accuracy-dashboard.sample.json')
  if (sample.ok && sample.headers.get('content-type')?.startsWith('application/json')) {
    return sample.json()
  }
  throw new Error('No dashboard data found. Run `npm run export:data`.')
}
