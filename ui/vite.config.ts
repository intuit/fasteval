import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Output goes into the Python package so it ships in the fasteval-core wheel.
// Stable file names (no hashes, one chunk) keep serving and the static
// report's inlining simple.
export default defineConfig({
  plugins: [react()],
  base: '/static/dist/',
  build: {
    outDir: '../fasteval/ui/static/dist',
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        entryFileNames: 'app.js',
        assetFileNames: (info) => (info.names?.[0]?.endsWith('.css') ? 'app.css' : '[name][extname]'),
        inlineDynamicImports: true,
      },
    },
  },
  server: {
    // `npm run dev` while `fasteval ui --port 8484` runs: hot reload + real API.
    proxy: { '/api': { target: 'http://localhost:8484', changeOrigin: true } },
  },
})
