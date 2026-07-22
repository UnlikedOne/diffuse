<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, default: '' },
  caption: { type: String, default: '' },
  unit: { type: String, default: '' },
  items: { type: Array, default: () => [] },
  max: { type: Number, default: 0 },
})

const maxValue = computed(() => {
  if (props.max) return props.max
  const top = Math.max(...props.items.map((i) => i.value))
  return top * 1.08
})

function width(value) {
  return Math.max(2, (value / maxValue.value) * 100)
}
</script>

<template>
  <figure class="dx-chart">
    <figcaption v-if="title" class="dx-chart__title">{{ title }}</figcaption>
    <div class="dx-chart__rows">
      <div v-for="item in items" :key="item.label" class="dx-chart__row">
        <span class="dx-chart__label">{{ item.label }}</span>
        <span class="dx-chart__track">
          <span
            class="dx-chart__fill"
            :class="item.tone || 'a'"
            :style="{ width: width(item.value) + '%' }"
          ></span>
        </span>
        <span class="dx-chart__value">{{ item.value }}{{ unit }}</span>
      </div>
    </div>
    <p v-if="caption" class="dx-chart__caption">{{ caption }}</p>
  </figure>
</template>
