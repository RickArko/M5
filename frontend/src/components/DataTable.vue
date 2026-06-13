<script setup>
import { compactNumber, modelLabel, percent, signed } from '../lib/format.js'

defineProps({
  rows: { type: Array, required: true },
  columns: { type: Array, required: true },
  limit: { type: Number, default: 25 },
})

function value(row, col) {
  const raw = row[col.key]
  if (col.key === 'model') return modelLabel(raw)
  if (col.kind === 'percent') return percent(raw)
  if (col.kind === 'signed') return signed(raw)
  if (typeof raw === 'number') return compactNumber(raw)
  return raw ?? 'n/a'
}
</script>

<template>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th v-for="col in columns" :key="col.key">{{ col.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows.slice(0, limit)" :key="JSON.stringify(row)">
          <td v-for="col in columns" :key="col.key">{{ value(row, col) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
