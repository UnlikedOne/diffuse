<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, default: '' },
  caption: { type: String, default: '' },
  unit: { type: String, default: '' },
  segments: { type: Array, default: () => [] },
})

const total = computed(() => props.segments.reduce((sum, seg) => sum + seg.value, 0))

function width(value) {
  return (value / total.value) * 100
}
</script>

<template>
  <figure class="dx-chart">
    <figcaption v-if="title" class="dx-chart__title">{{ title }}</figcaption>
    <div class="dx-stack">
      <span
        v-for="seg in segments"
        :key="seg.label"
        class="dx-stack__seg"
        :class="seg.tone || 'a'"
        :style="{ width: width(seg.value) + '%' }"
      >{{ Math.round(width(seg.value)) }}%</span>
    </div>
    <div class="dx-stack__legend">
      <span v-for="seg in segments" :key="seg.label" class="dx-stack__key">
        <i class="dx-stack__dot" :class="seg.tone || 'a'"></i>
        {{ seg.label }} · {{ seg.value }}{{ unit }}
      </span>
    </div>
    <p v-if="caption" class="dx-chart__caption">{{ caption }}</p>
  </figure>
</template>
