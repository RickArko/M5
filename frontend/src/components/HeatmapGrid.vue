<script setup>
import { computed } from 'vue'

import { compactNumber, modelLabel } from '../lib/format.js'

const props = defineProps({
  rows: { type: Array, required: true },
  metric: { type: String, required: true },
})

const models = computed(() => [...new Set(props.rows.map((row) => row.model))])
const levels = computed(() =>
  [...new Set(props.rows.map((row) => row.level))]
    .map((level) => props.rows.find((row) => row.level === level))
    .sort((a, b) => Number(a.level_idx) - Number(b.level_idx))
    .map((row) => row.level),
)
const values = computed(() => props.rows.map((row) => Number(row[props.metric])).filter(Number.isFinite))
const min = computed(() => Math.min(...values.value, 0))
const max = computed(() => Math.max(...values.value, 1))

function cell(model, level) {
  return props.rows.find((row) => row.model === model && row.level === level)
}

function intensity(value) {
  const ratio = (Number(value) - min.value) / Math.max(max.value - min.value, 0.001)
  return 1 - ratio
}
</script>

<template>
  <div class="heatmap">
    <div class="heatmap-head"></div>
    <div v-for="level in levels" :key="level" class="heatmap-level" :title="level">{{ level }}</div>
    <template v-for="model in models" :key="model">
      <div class="heatmap-model" :title="modelLabel(model)">{{ modelLabel(model) }}</div>
      <div
        v-for="level in levels"
        :key="`${model}-${level}`"
        class="heatmap-cell"
        :style="{ '--heat': intensity(cell(model, level)?.[metric] ?? max) }"
      >
        {{ cell(model, level) ? compactNumber(cell(model, level)[metric]) : 'n/a' }}
      </div>
    </template>
  </div>
</template>
