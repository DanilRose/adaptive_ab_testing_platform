// frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,          // слушать на 0.0.0.0 внутри Docker-контейнера
    watch: {
      usePolling: true,  // polling нужен для Windows + Docker (WSL2/bind mounts)
    },
    proxy: {
      '/api': {
        target: 'http://backend:8000', // имя сервиса из docker-compose
        changeOrigin: true
      }
    }
  }
})