<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'

const canvas = ref<HTMLCanvasElement | null>(null)
let raf = 0
let ro: ResizeObserver | null = null

type Node = { x: number; y: number; vx: number; vy: number; r: number }

onMounted(() => {
  const el = canvas.value
  if (!el) return
  const ctx = el.getContext('2d')
  if (!ctx) return

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  let W = 0
  let H = 0
  let dpr = Math.min(window.devicePixelRatio || 1, 2)
  let nodes: Node[] = []

  const build = () => {
    const rect = el.getBoundingClientRect()
    W = rect.width
    H = rect.height
    dpr = Math.min(window.devicePixelRatio || 1, 2)
    el.width = Math.round(W * dpr)
    el.height = Math.round(H * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const count = Math.max(14, Math.min(46, Math.round((W * H) / 16000)))
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.28,
      vy: (Math.random() - 0.5) * 0.28,
      r: 1.4 + Math.random() * 1.8,
    }))
  }

  const LINK = 132
  let t = 0

  const draw = () => {
    ctx.clearRect(0, 0, W, H)
    t += 0.012

    for (const n of nodes) {
      n.x += n.vx
      n.y += n.vy
      if (n.x < 0 || n.x > W) n.vx *= -1
      if (n.y < 0 || n.y > H) n.vy *= -1
    }

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i]
        const b = nodes[j]
        const dx = a.x - b.x
        const dy = a.y - b.y
        const d = Math.hypot(dx, dy)
        if (d < LINK) {
          const alpha = (1 - d / LINK) * 0.5
          const mid = (Math.sin(t + i * 0.6) + 1) / 2
          const g = ctx.createLinearGradient(a.x, a.y, b.x, b.y)
          g.addColorStop(0, `rgba(45, 212, 191, ${alpha})`)
          g.addColorStop(mid, `rgba(34, 211, 238, ${alpha})`)
          g.addColorStop(1, `rgba(139, 92, 246, ${alpha})`)
          ctx.strokeStyle = g
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(b.x, b.y)
          ctx.stroke()
        }
      }
    }

    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i]
      const pulse = 0.6 + 0.4 * Math.sin(t * 1.6 + i)
      ctx.beginPath()
      ctx.arc(n.x, n.y, n.r + pulse * 1.1, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(94, 234, 212, ${0.55 + pulse * 0.35})`
      ctx.fill()
      ctx.beginPath()
      ctx.arc(n.x, n.y, n.r + 6, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(34, 211, 238, ${0.06 + pulse * 0.05})`
      ctx.fill()
    }

    if (!reduce) raf = requestAnimationFrame(draw)
  }

  build()
  draw()

  ro = new ResizeObserver(() => {
    build()
    if (reduce) draw()
  })
  ro.observe(el)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  ro?.disconnect()
})
</script>

<template>
  <section class="dx-net" aria-hidden="true">
    <canvas ref="canvas" class="dx-net__canvas"></canvas>
    <div class="dx-net__overlay">
      <p class="dx-net__kicker">Peer to peer</p>
      <h2 class="dx-net__title">A network that heals itself</h2>
      <p class="dx-net__sub">Nodes find each other, share the load, and close the gap when one drops. No master, no server.</p>
    </div>
  </section>
</template>
