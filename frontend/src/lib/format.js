export function compactNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  return new Intl.NumberFormat('en-US', {
    notation: Math.abs(Number(value)) >= 10000 ? 'compact' : 'standard',
    maximumFractionDigits: digits,
  }).format(Number(value))
}

export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function signed(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a'
  const n = Number(value)
  return `${n > 0 ? '+' : ''}${compactNumber(n, digits)}`
}

export function modelLabel(model) {
  const label = String(model).replaceAll('_', ' / ')
  if (label === 'toto / TOTO') return 'TOTO'
  if (label === 'lgbm / LGBM') return 'LightGBM'
  return label
}
