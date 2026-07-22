import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import DxBars from './components/DxBars.vue'
import DxStack from './components/DxStack.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('DxBars', DxBars)
    app.component('DxStack', DxStack)
  },
} satisfies Theme
