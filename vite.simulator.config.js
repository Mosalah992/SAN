import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  root: path.join(dir, "frontend", "simulator"),
  base: "/static/simulator/",
  build: {
    outDir: path.join(dir, "frontend", "siem", "simulator"),
    emptyOutDir: true,
  },
});
