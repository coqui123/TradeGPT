import { type Config } from "tailwindcss";

export default {
  content: [
    "{routes,islands,components}/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        roboto: ["Roboto", "sans-serif"],
        montserrat: ["Montserrat", "sans-serif"],
      },
      colors: {
        primary: {
          DEFAULT: "#1e3a8a",
          dark: "#1e3a8a",
        },
        secondary: "#f0f9ff",
        accent: {
          DEFAULT: "#3b82f6",
          hover: "#2563eb",
        },
        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        neutral: "#6b7280",
      },
    },
  },
} satisfies Config;
