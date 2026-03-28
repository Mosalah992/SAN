/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["frontend/siem/**/*.{html,js}"],
  theme: {
    extend: {
      colors: {
        /* Semantic tokens: --primary 142 70% 45%, --glow-cyan 185 70% 50%, --glow-amber 38 92% 50% */
        primary: "hsl(142 70% 45%)",
        "glow-cyan": "hsl(185 70% 50%)",
        "glow-amber": "hsl(38 92% 50%)",
        cyber: {
          black: "#000000",
          void: "hsl(150 40% 1%)",
          surface: "hsl(150 30% 2%)",
          green: "hsl(142 70% 45%)",
          "green-dim": "hsl(142 70% 35%)",
          "green-muted": "hsl(150 50% 20%)",
          cyan: "hsl(185 70% 50%)",
          "cyan-dim": "hsl(185 70% 40%)",
          yellow: "hsl(70 100% 50%)",
          red: "hsl(340 100% 50%)",
          magenta: "hsl(300 100% 50%)",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Courier New"', "monospace"],
        sans: ['"Space Grotesk"', "system-ui", "sans-serif"],
      },
      boxShadow: {
        "glow-green": "0 0 12px hsl(142 70% 45% / 0.4)",
        "glow-green-sm": "0 0 6px hsl(142 70% 45% / 0.3)",
        "glow-cyan": "0 0 12px hsl(185 70% 50% / 0.4)",
        "glow-cyan-sm": "0 0 6px hsl(185 70% 50% / 0.3)",
        "glow-yellow": "0 0 12px hsl(70 100% 50% / 0.35)",
        "glow-red": "0 0 12px hsl(340 100% 50% / 0.4)",
      },
      animation: {
        "glow-pulse": "glow-pulse 2s ease-in-out infinite",
        "scanline": "scanline 4s linear infinite",
      },
      keyframes: {
        "glow-pulse": {
          "0%, 100%": { opacity: "1", filter: "brightness(1)" },
          "50%": { opacity: "0.85", filter: "brightness(1.1)" },
        },
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
      },
    },
  },
  plugins: [],
};
