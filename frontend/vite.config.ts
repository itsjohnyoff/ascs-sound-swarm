import { defineConfig } from "vite";
import { resolve } from "node:path";

// Dev: `npm run dev` on :5173, proxying API + sockets to the FastAPI server on :8000
// (`uvicorn backend.server.app:app`). Prod: FastAPI serves dist/ itself.
const target = process.env.ASCS_SERVER ?? "http://localhost:8000";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        phone: resolve(__dirname, "phone.html"),
        chirp: resolve(__dirname, "chirp.html"),
      },
    },
  },
  server: {
    proxy: {
      "/api": { target, changeOrigin: true },
      "/ws": { target, ws: true, changeOrigin: true },
    },
  },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
