import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendUrl = process.env.VITE_MOA_BACKEND_URL ?? "http://127.0.0.1:8008";
const frontendPort = Number(process.env.VITE_MOA_FRONTEND_PORT ?? 5173);

export default defineConfig({
  plugins: [react()],
  server: {
    port: frontendPort,
    proxy: {
      "/api": backendUrl
    }
  },
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        activity: "activity.html"
      }
    }
  }
});
