// Vite configuration for the SilverSight frontend.
// Dev: `npm run dev` serves on http://localhost:5173 with hot reload and forwards
// /api/* to the FastAPI backend on port 8000 (start it with `xag ui --reload`).
// Build: `npm run build` writes static files to web/dist, which FastAPI serves.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
