<script setup lang="ts">
const year = new Date().getFullYear()
</script>

<template>
  <footer class="dx-footer">
    <div class="dx-footer__inner">
      <div class="dx-footer__brand">
        <img class="dx-footer__logo" src="/diffuse/logo.svg" alt="" aria-hidden="true" />
        <div>
          <p class="dx-footer__name">Diffuse</p>
          <p class="dx-footer__tag">Your own AI, split across the world, watched by no one.</p>
          <div class="dx-footer__badges">
            <span>no servers</span>
            <span>no surveillance</span>
            <span>no prompt logging</span>
          </div>
        </div>
      </div>

      <nav class="dx-footer__cols">
        <div class="dx-footer__col">
          <p class="dx-footer__head">Start</p>
          <a href="/diffuse/start/installation">Installation</a>
          <a href="/diffuse/start/quickstart">Quickstart</a>
          <a href="/diffuse/guides/chat">Chat with the network</a>
        </div>
        <div class="dx-footer__col">
          <p class="dx-footer__head">Learn</p>
          <a href="/diffuse/introduction/how-it-works">How it works</a>
          <a href="/diffuse/concepts/pipeline">Concepts</a>
          <a href="/diffuse/benchmarks">Benchmarks</a>
        </div>
        <div class="dx-footer__col">
          <p class="dx-footer__head">Reference</p>
          <a href="/diffuse/reference/cli">CLI</a>
          <a href="/diffuse/reference/api">OpenAI API</a>
          <a href="/diffuse/privacy">Privacy</a>
        </div>
        <div class="dx-footer__col">
          <p class="dx-footer__head">Project</p>
          <a href="https://github.com/UnlikedOne/diffuse" target="_blank" rel="noopener">GitHub</a>
          <a href="https://github.com/UnlikedOne/diffuse/releases" target="_blank" rel="noopener">Changelog</a>
          <a href="/diffuse/limitations">Limitations</a>
        </div>
      </nav>
    </div>

    <div class="dx-footer__bar">
      <span>&copy; {{ year }} Diffuse &middot; Released under the AGPL-3.0 License.</span>
      <span class="dx-footer__pulse"><i></i> peer-to-peer, no logging</span>
    </div>
  </footer>
</template>
