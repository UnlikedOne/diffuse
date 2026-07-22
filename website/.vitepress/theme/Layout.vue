<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import DxFooter from './components/DxFooter.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #layout-top>
      <div class="dx-bg" aria-hidden="true">
        <div class="dx-bg__aurora"></div>
        <div class="dx-bg__grid"></div>
      </div>
    </template>
    <template #layout-bottom>
      <DxFooter />
    </template>
  </Layout>
</template>
