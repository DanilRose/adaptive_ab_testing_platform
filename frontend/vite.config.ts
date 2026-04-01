// frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const proxyTarget = process.env.FRONTEND_PROXY_TARGET || 'http://backend:8000'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['lodash/get', 'lodash/set', 'lodash/throttle', 'lodash/cloneDeep'],
  },
  server: {
    port: 3000,
    host: true,
    watch: {
      usePolling: true,
    },
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
