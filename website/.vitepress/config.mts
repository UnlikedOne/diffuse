import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Diffuse',
  description: 'Distributed peer-to-peer LLM inference where your prompt never leaves your device.',
  lang: 'en-US',
  base: process.env.DEPLOY_BASE || '/',
  cleanUrls: true,
  appearance: 'dark',
  head: [
    ['meta', { name: 'theme-color', content: '#0a0b0f' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Diffuse' }],
    ['meta', { property: 'og:description', content: 'Your own AI, split across the world, watched by no one.' }],
  ],
  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Diffuse',
    outline: { level: [2, 3], label: 'On this page' },
    search: {
      provider: 'local',
    },
    nav: [
      { text: 'Introduction', link: '/introduction/what-is-diffuse' },
      { text: 'Get started', link: '/start/installation' },
      { text: 'Guides', link: '/guides/chat' },
      { text: 'Reference', link: '/reference/cli' },
      { text: 'Benchmarks', link: '/benchmarks' },
      { text: 'FAQ', link: '/faq' },
      { text: 'Privacy', link: '/privacy' },
      {
        text: 'v0.2.7',
        items: [
          { text: 'Changelog', link: 'https://github.com/UnlikedOne/diffuse/releases' },
          { text: 'Limitations', link: '/limitations' },
        ],
      },
    ],
    sidebar: [
      {
        text: 'Introduction',
        collapsed: false,
        items: [
          { text: 'What is Diffuse', link: '/introduction/what-is-diffuse' },
          { text: 'How it works', link: '/introduction/how-it-works' },
          { text: 'Use cases', link: '/use-cases' },
          { text: 'How Diffuse compares', link: '/comparison' },
        ],
      },
      {
        text: 'Get started',
        collapsed: false,
        items: [
          { text: 'Installation', link: '/start/installation' },
          { text: 'Quickstart', link: '/start/quickstart' },
        ],
      },
      {
        text: 'Concepts',
        collapsed: false,
        items: [
          { text: 'Slices and the pipeline', link: '/concepts/pipeline' },
          { text: 'Gossip and discovery', link: '/concepts/gossip' },
          { text: 'Replication and healing', link: '/concepts/replication' },
          { text: 'NAT relay', link: '/concepts/nat-relay' },
          { text: 'Trust and encryption', link: '/concepts/trust' },
        ],
      },
      {
        text: 'Guides',
        collapsed: false,
        items: [
          { text: 'Chat with the network', link: '/guides/chat' },
          { text: 'Ask one question', link: '/guides/query' },
          { text: 'Host a node', link: '/guides/host' },
          { text: 'OpenAI-compatible server', link: '/guides/server' },
          { text: 'Choosing a model', link: '/guides/choosing-a-model' },
          { text: 'Self-hosting a network', link: '/self-hosting' },
        ],
      },
      {
        text: 'Reference',
        collapsed: false,
        items: [
          { text: 'CLI', link: '/reference/cli' },
          { text: 'OpenAI API', link: '/reference/api' },
        ],
      },
      {
        text: 'Performance',
        collapsed: false,
        items: [
          { text: 'Benchmarks', link: '/benchmarks' },
        ],
      },
      {
        text: 'Trust',
        collapsed: false,
        items: [
          { text: 'Privacy and threat model', link: '/privacy' },
          { text: 'Troubleshooting', link: '/troubleshooting' },
          { text: 'Limitations', link: '/limitations' },
        ],
      },
      {
        text: 'Resources',
        collapsed: false,
        items: [
          { text: 'FAQ', link: '/faq' },
          { text: 'Glossary', link: '/glossary' },
          { text: 'Roadmap', link: '/roadmap' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/UnlikedOne/diffuse' },
    ],
    editLink: {
      pattern: 'https://github.com/UnlikedOne/diffuse/edit/main/website/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
