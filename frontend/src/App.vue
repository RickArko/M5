<script setup>
import { computed, onMounted, ref } from 'vue'

import DataTable from './components/DataTable.vue'
import HeatmapGrid from './components/HeatmapGrid.vue'
import KpiCard from './components/KpiCard.vue'
import LineChart from './components/LineChart.vue'
import MetricBarChart from './components/MetricBarChart.vue'
import { availableValues, bestRow, filterRows, loadDashboardData, METRICS, metricMeta, rankRows } from './lib/dashboard.js'
import { compactNumber, modelLabel, percent } from './lib/format.js'

const data = ref(null)
const loadError = ref('')
const selectedMetric = ref('wrmsse')
const selectedModel = ref('')
const selectedLevel = ref('')
const selectedSegmentAxis = ref('state_id')
const selectedSegment = ref('')

onMounted(async () => {
  try {
    data.value = await loadDashboardData()
    selectedModel.value = bestRow(data.value.headline, 'wrmsse')?.model ?? ''
  } catch (error) {
    loadError.value = error.message
  }
})

const metric = computed(() => metricMeta(selectedMetric.value))
const headline = computed(() => rankRows(data.value?.headline ?? [], selectedMetric.value))
const best = computed(() => bestRow(data.value?.headline ?? [], 'wrmsse'))
const modelOptions = computed(() => availableValues(data.value?.headline ?? [], 'model'))
const levelOptions = computed(() => availableValues(data.value?.levels ?? [], 'level'))
const segmentAxisOptions = computed(() => availableValues(data.value?.segments ?? [], 'segment_axis'))
const segmentOptions = computed(() =>
  availableValues((data.value?.segments ?? []).filter((row) => row.segment_axis === selectedSegmentAxis.value), 'segment'),
)
const visibleLevels = computed(() =>
  filterRows(data.value?.levels ?? [], {
    model: selectedModel.value || '',
    level: selectedLevel.value || '',
  }),
)
const visibleSegments = computed(() =>
  rankRows(
    filterRows(data.value?.segments ?? [], {
      model: selectedModel.value || '',
      segment_axis: selectedSegmentAxis.value,
      segment: selectedSegment.value || '',
    }),
    selectedMetric.value,
  ),
)
const visibleFva = computed(() =>
  (data.value?.fva ?? []).filter((row) => !selectedModel.value || row.model === selectedModel.value),
)
const chartModels = computed(() => {
  const chosen = selectedModel.value ? [selectedModel.value] : []
  const top = headline.value.slice(0, 4).map((row) => row.model)
  return [...new Set([...chosen, ...top])]
})
const latestCumError = computed(() => {
  const rows = (data.value?.cumulative_error ?? []).filter((row) => row.model === selectedModel.value)
  return rows.sort((a, b) => Number(b.h) - Number(a.h))[0] ?? null
})
const selectedHeadline = computed(() =>
  (data.value?.headline ?? []).find((row) => row.model === selectedModel.value) ?? best.value,
)

function metricDisplay(value) {
  return metric.value.percent ? percent(value) : compactNumber(value)
}
</script>

<template>
  <main>
    <section class="hero">
      <div>
        <p class="eyebrow">M5 Accuracy Observatory</p>
        <h1>Forecast performance by model, hierarchy, segment, and horizon.</h1>
        <p class="lede">
          Compare WRMSSE, Bias, MAE, sMAPE, wMAPE, FVA, and cumulative horizon error from the
          repo's rolling-origin CV artifacts.
        </p>
      </div>
      <aside v-if="data" class="source-card">
        <span>Generated</span>
        <strong>{{ data.generated_at }}</strong>
        <small>{{ data.source.n_series }} series · {{ data.source.n_rows }} CV rows</small>
      </aside>
    </section>

    <section v-if="loadError" class="panel error">{{ loadError }}</section>

    <template v-if="data">
      <section class="controls panel">
        <label>
          Metric
          <select v-model="selectedMetric">
            <option v-for="item in METRICS" :key="item.key" :value="item.key">{{ item.label }}</option>
          </select>
        </label>
        <label>
          Focus model
          <select v-model="selectedModel">
            <option value="">Top models</option>
            <option v-for="model in modelOptions" :key="model" :value="model">{{ modelLabel(model) }}</option>
          </select>
        </label>
        <label>
          Hierarchy level
          <select v-model="selectedLevel">
            <option value="">All levels</option>
            <option v-for="level in levelOptions" :key="level" :value="level">{{ level }}</option>
          </select>
        </label>
        <label>
          Segment axis
          <select v-model="selectedSegmentAxis" @change="selectedSegment = ''">
            <option v-for="axis in segmentAxisOptions" :key="axis" :value="axis">{{ axis }}</option>
          </select>
        </label>
        <label>
          Segment
          <select v-model="selectedSegment">
            <option value="">All {{ selectedSegmentAxis }}</option>
            <option v-for="segment in segmentOptions" :key="segment" :value="segment">{{ segment }}</option>
          </select>
        </label>
      </section>

      <section class="kpi-grid">
        <KpiCard label="Best WRMSSE" :value="best?.wrmsse" :detail="modelLabel(best?.model ?? '')" />
        <KpiCard label="Focus MAE" :value="selectedHeadline?.mae" :detail="modelLabel(selectedHeadline?.model ?? '')" />
        <KpiCard label="Focus Bias %" :value="selectedHeadline?.bias_pct" mode="percent" detail="signed forecast error / actual" />
        <KpiCard
          label="Cumulative Error"
          :value="latestCumError?.cum_error_pct"
          mode="percent"
          :detail="latestCumError ? `through h${latestCumError.h}` : 'select a model'"
          tone="warm"
        />
      </section>

      <section class="grid two">
        <article class="panel">
          <div class="panel-title">
            <div>
              <span>Leaderboard</span>
              <h2>{{ metric.label }} ranking</h2>
            </div>
            <strong>{{ metricDisplay(headline[0]?.[selectedMetric]) }}</strong>
          </div>
          <MetricBarChart :rows="headline" :metric="selectedMetric" />
        </article>

        <article class="panel">
          <div class="panel-title">
            <div>
              <span>Forecast Value Added</span>
              <h2>Value vs baseline</h2>
            </div>
          </div>
          <DataTable
            :rows="visibleFva"
            :limit="8"
            :columns="[
              { key: 'model', label: 'Model' },
              { key: 'baseline', label: 'Baseline' },
              { key: 'fva_abs', label: 'FVA', kind: 'signed' },
              { key: 'fva_pct', label: 'FVA %', kind: 'percent' },
            ]"
          />
        </article>
      </section>

      <section class="panel">
        <div class="panel-title">
          <div>
            <span>Aggregation Levels</span>
            <h2>Heatmap across the 12 M5 hierarchy levels</h2>
          </div>
        </div>
        <HeatmapGrid :rows="data.levels" :metric="selectedMetric" />
      </section>

      <section class="grid two">
        <article class="panel">
          <div class="panel-title">
            <div>
              <span>Horizon Degradation</span>
              <h2>WRMSSE by lead day</h2>
            </div>
          </div>
          <LineChart :rows="data.horizon" y-key="wrmsse" :selected-groups="chartModels" />
        </article>

        <article class="panel">
          <div class="panel-title">
            <div>
              <span>Cumulative Forecast Error</span>
              <h2>Signed error over horizon</h2>
            </div>
          </div>
          <LineChart :rows="data.cumulative_error" y-key="cum_error_pct" :selected-groups="chartModels" />
        </article>
      </section>

      <section class="grid two">
        <article class="panel">
          <div class="panel-title">
            <div>
              <span>Drill Down</span>
              <h2>Selected aggregation metrics</h2>
            </div>
          </div>
          <DataTable
            :rows="visibleLevels"
            :columns="[
              { key: 'level', label: 'Level' },
              { key: 'n_series', label: 'Series' },
              { key: 'wrmsse', label: 'WRMSSE' },
              { key: 'mae', label: 'MAE' },
              { key: 'wmape', label: 'wMAPE', kind: 'percent' },
              { key: 'bias_pct', label: 'Bias %', kind: 'percent' },
            ]"
          />
        </article>

        <article class="panel">
          <div class="panel-title">
            <div>
              <span>Segment Zoom</span>
              <h2>{{ selectedSegmentAxis }} groups</h2>
            </div>
          </div>
          <DataTable
            :rows="visibleSegments"
            :columns="[
              { key: 'segment', label: 'Segment' },
              { key: 'n_series', label: 'Series' },
              { key: 'wrmsse', label: 'WRMSSE' },
              { key: 'mae', label: 'MAE' },
              { key: 'smape', label: 'sMAPE', kind: 'percent' },
              { key: 'bias', label: 'Bias', kind: 'signed' },
            ]"
          />
        </article>
      </section>
    </template>
  </main>
</template>
