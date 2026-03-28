import { defineConfig } from "vite";
import path from "path";
import { fileURLToPath } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const siemDir = path.join(dir, "frontend", "siem");

export default defineConfig({
  root: siemDir,
  base: "/",
  build: {
    outDir: path.join(siemDir, "dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: path.join(siemDir, "index.html"),
    },
  },
  server: {
    port: 5174,
    proxy: {
      "/api": { target: "http://127.0.0.1:8787", changeOrigin: true },
      "/sounds": { target: "http://127.0.0.1:8787", changeOrigin: true },
      "/simulator": { target: "http://127.0.0.1:8787", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8787", ws: true },
    },
  },
});
