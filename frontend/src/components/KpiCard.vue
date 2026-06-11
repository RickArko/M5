<script setup>
import { compactNumber, percent, signed } from '../lib/format.js'

defineProps({
  label: { type: String, required: true },
  value: { type: Number, default: null },
  detail: { type: String, default: '' },
  mode: { type: String, default: 'number' },
  tone: { type: String, default: 'neutral' },
})

function display(value, mode) {
  if (mode === 'percent') return percent(value)
  if (mode === 'signed') return signed(value)
  return compactNumber(value)
}
</script>

<template>
  <article class="kpi-card" :class="`tone-${tone}`">
    <span>{{ label }}</span>
    <strong>{{ display(value, mode) }}</strong>
    <small>{{ detail }}</small>
  </article>
</template>
