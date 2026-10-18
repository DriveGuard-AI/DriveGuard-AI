import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时前端由 Vite 提供，并通过代理访问后端 REST / WebSocket；
// 生产构建产物由 FastAPI 托管（见 backend/app/main.py）。
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
})
