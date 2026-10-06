import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Local dev: forward API calls to the FastAPI backend.
    proxy: {
      '/api': 'http://localhost:7860',
      '/health': 'http://localhost:7860',
    },
  },
})
