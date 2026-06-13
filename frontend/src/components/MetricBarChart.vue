<script setup>
import { computed } from 'vue'

import { compactNumber, modelLabel } from '../lib/format.js'

const props = defineProps({
  rows: { type: Array, required: true },
  metric: { type: String, required: true },
  limit: { type: Number, default: 10 },
})

const ranked = computed(() => props.rows.slice(0, props.limit))
const maxValue = computed(() => Math.max(...ranked.value.map((row) => Math.abs(Number(row[props.metric]) || 0)), 0.001))
</script>

<template>
  <div class="bar-chart">
    <div v-for="row in ranked" :key="row.model + row.segment + row.level" class="bar-row">
      <div class="bar-label" :title="modelLabel(row.model)">{{ modelLabel(row.model) }}</div>
      <div class="bar-track">
        <div class="bar-fill" :style="{ width: `${(Math.abs(row[metric]) / maxValue) * 100}%` }"></div>
      </div>
      <div class="bar-value">{{ compactNumber(row[metric]) }}</div>
    </div>
  </div>
</template>
