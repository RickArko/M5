import { describe, expect, it } from 'vitest'

import { availableValues, bestRow, filterRows, rankRows } from './dashboard.js'

const rows = [
  { model: 'a', level: 'Total', segment_axis: 'state_id', segment: 'CA', wrmsse: 0.8, bias: 4 },
  { model: 'b', level: 'Total', segment_axis: 'state_id', segment: 'TX', wrmsse: 0.6, bias: -1 },
  { model: 'c', level: 'state_id', segment_axis: 'cat_id', segment: 'FOODS', wrmsse: 0.7, bias: 2 },
]

describe('dashboard helpers', () => {
  it('ranks lower-is-better metrics ascending', () => {
    expect(rankRows(rows, 'wrmsse').map((row) => row.model)).toEqual(['b', 'c', 'a'])
  })

  it('selects bias by smallest absolute value', () => {
    expect(bestRow(rows, 'bias').model).toBe('b')
  })

  it('filters by the active drill-down dimensions', () => {
    expect(filterRows(rows, { segment_axis: 'state_id', segment: 'CA' })).toHaveLength(1)
  })

  it('returns sorted distinct values', () => {
    expect(availableValues(rows, 'segment_axis')).toEqual(['cat_id', 'state_id'])
  })
})
