import { type PageProps } from "$fresh/server.ts";

export default function App({ Component }: PageProps) {
  return (
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>TradeGPT - AI-Powered Crypto Trading Analysis</title>
        <link rel="stylesheet" href="/styles.css" />
        <link
          href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Montserrat:wght@400;600;700&display=swap"
          rel="stylesheet"
        />
        <link
          rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
        />
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js" defer></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js" defer></script>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js" defer></script>
      </head>
      <body class="font-roboto bg-[#f8fafc] text-[#1e293b]">
        <Component />
      </body>
    </html>
  );
}
