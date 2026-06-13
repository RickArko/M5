<script setup>
import { computed, ref } from 'vue'

import { compactNumber, modelLabel, percent } from '../lib/format.js'

const props = defineProps({
  rows: { type: Array, required: true },
  xKey: { type: String, default: 'h' },
  yKey: { type: String, required: true },
  groupKey: { type: String, default: 'model' },
  selectedGroups: { type: Array, default: () => [] },
})

const width = 720
const height = 260
const pad = 34
const colors = ['#e84d2f', '#0f8b8d', '#f2a541', '#355070', '#7d5fff', '#3a86ff', '#2d6a4f']
const hovered = ref(null)

const groups = computed(() => {
  const allowed = new Set(props.selectedGroups)
  const map = new Map()
  for (const row of props.rows) {
    if (allowed.size && !allowed.has(row[props.groupKey])) continue
    if (!map.has(row[props.groupKey])) map.set(row[props.groupKey], [])
    map.get(row[props.groupKey]).push(row)
  }
  for (const values of map.values()) values.sort((a, b) => Number(a[props.xKey]) - Number(b[props.xKey]))
  return [...map.entries()]
})

const domain = computed(() => {
  const points = groups.value.flatMap(([, rows]) => rows)
  const xs = points.map((row) => Number(row[props.xKey]))
  const ys = points.map((row) => Number(row[props.yKey]))
  return {
    minX: Math.min(...xs, 0),
    maxX: Math.max(...xs, 1),
    minY: Math.min(...ys, 0),
    maxY: Math.max(...ys, 1),
  }
})

function sx(x) {
  const d = domain.value
  return pad + ((Number(x) - d.minX) / Math.max(d.maxX - d.minX, 1)) * (width - pad * 2)
}

function sy(y) {
  const d = domain.value
  return height - pad - ((Number(y) - d.minY) / Math.max(d.maxY - d.minY, 0.001)) * (height - pad * 2)
}

function path(rows) {
  return rows.map((row, i) => `${i === 0 ? 'M' : 'L'}${sx(row[props.xKey])},${sy(row[props.yKey])}`).join(' ')
}

function color(index) {
  return colors[index % colors.length]
}

function formatValue(value) {
  return props.yKey.includes('pct') ? percent(value) : compactNumber(value, 3)
}

function showTooltip(point, name, index) {
  const x = sx(point[props.xKey])
  const y = sy(point[props.yKey])
  hovered.value = {
    x,
    y,
    tx: Math.min(Math.max(x + 14, 44), width - 224),
    ty: Math.max(y - 64, 12),
    color: color(index),
    group: modelLabel(name),
    xLabel: `${props.xKey} ${point[props.xKey]}`,
    yLabel: formatValue(point[props.yKey]),
  }
}
</script>

<template>
  <div class="line-chart-wrap">
    <svg class="line-chart" viewBox="0 0 720 260" role="img" @mouseleave="hovered = null">
      <line :x1="pad" :x2="width - pad" :y1="height - pad" :y2="height - pad" class="axis" />
      <line :x1="pad" :x2="pad" :y1="pad" :y2="height - pad" class="axis" />
      <path
        v-for="([name, values], index) in groups"
        :key="name"
        :d="path(values)"
        :stroke="color(index)"
        class="line"
      />
      <g v-for="([name, values], index) in groups" :key="`${name}-points`">
        <circle
          v-for="point in values"
          :key="`${name}-${point[xKey]}`"
          :cx="sx(point[xKey])"
          :cy="sy(point[yKey])"
          r="8"
          class="hit-point"
          @focus="showTooltip(point, name, index)"
          @mouseenter="showTooltip(point, name, index)"
        />
        <circle
          v-for="point in values"
          :key="`${name}-${point[xKey]}-dot`"
          :cx="sx(point[xKey])"
          :cy="sy(point[yKey])"
          r="4"
          :fill="color(index)"
          class="line-dot"
        />
      </g>
      <g v-if="hovered" class="chart-tooltip">
        <line :x1="hovered.x" :x2="hovered.x" :y1="pad" :y2="height - pad" class="hover-line" />
        <circle :cx="hovered.x" :cy="hovered.y" r="6" :fill="hovered.color" class="hover-dot" />
        <rect :x="hovered.tx" :y="hovered.ty" width="210" height="58" rx="14" />
        <circle :cx="hovered.tx + 17" :cy="hovered.ty + 19" r="5" :fill="hovered.color" />
        <text :x="hovered.tx + 30" :y="hovered.ty + 23" class="tooltip-title">{{ hovered.group }}</text>
        <text :x="hovered.tx + 16" :y="hovered.ty + 45" class="tooltip-meta">
          {{ hovered.xLabel }} · {{ hovered.yLabel }}
        </text>
      </g>
    </svg>
    <div class="chart-legend" aria-label="Chart legend">
      <span v-for="([name], index) in groups" :key="`${name}-legend`">
        <i :style="{ background: color(index) }"></i>
        {{ modelLabel(name) }}
      </span>
    </div>
  </div>
</template>
