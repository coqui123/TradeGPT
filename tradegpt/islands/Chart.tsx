import { useSignal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";

declare global {
  interface Window {
    echarts: any;
  }
}

const API_BASE_URL = "http://localhost:8001";

interface ChartProps {
  initialPair?: string;
}

interface AnalysisResult {
  pair: string;
  signal: "BUY" | "SELL";
  trade_size: string;
  take_profit: {
    tp1: string;
    tp2: string;
    tp3: string;
  };
  stop_loss: {
    initial: string;
    breakeven: string;
    trailing: string;
  };
  confidence: "HIGH" | "MEDIUM" | "LOW";
  explanation: string;
  timeframe: string;
  key_levels: {
    support_levels: Array<{
      price: number;
      strength: string;
      type: string;
    }>;
    resistance_levels: Array<{
      price: number;
      strength: string;
      type: string;
    }>;
    targets: number[];
  };
  analysis_details: {
    technical_indicators: Record<string, any>;
    market_summary: {
      price_summary: {
        last_price: number;
        price_high_24h: number;
        price_low_24h: number;
        price_average_24h: number;
        price_volatility: number;
      };
      volume_summary: {
        volume_average_24h: number;
        volume_highest_24h: number;
        volume_lowest_24h: number;
      };
      market_sentiment: {
        buy_sell_ratio: number | null;
        dominant_side: "BUY" | "SELL";
      };
    };
    current_price: number;
    amount_usd: number;
  };
}

interface Order {
  id: string;
  product_id: string;
  side: string;
  status: string;
  order_type: string;
  base_size: string;
  quote_size?: string;
  limit_price?: string;
  created_time: string;
  filled_size: string;
}

interface Candle {
  start: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ChartClickParam {
  time: number;
  point: {
    x: number;
    y: number;
  };
}

interface ChartCrosshairParam {
  time?: number;
  point?: {
    x: number;
    y: number;
  };
}

interface ChartScale {
  max: number;
  min: number;
}

interface ChartInstance {
  legend: {
    afterFit: () => void;
    height: number;
  };
  ctx: CanvasRenderingContext2D;
  scales: {
    x: any;
    y: any;
  };
}

interface ChartContext {
  chart: ChartInstance;
  dataset: {
    data: any[];
    borderColor?: string;
    borderWidth?: number;
    type?: string;
  };
  raw: {
    x?: number;
    y?: number;
    o?: number;
    h?: number;
    l?: number;
    c?: number;
  };
}

interface CandleDataPoint {
  x: number;
  o: number;
  h: number;
  l: number;
  c: number;
}

interface IndicatorSettings {
  keyLevels: boolean;
  movingAverages: boolean;
  rsi: boolean;
  volume: boolean;
  macd: boolean;
  bollinger: boolean;
  ichimoku: boolean;
}

function OrderList({ orders }: { orders: Order[] }) {
  if (!orders.length) {
    return (
      <div class="text-center text-gray-500 py-8">
        No orders found
      </div>
    );
  }

  return (
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
      {orders.map((order) => (
        <div key={order.id} class="bg-white p-4 rounded-lg shadow">
          <div class="flex justify-between items-center mb-2">
            <span class="font-semibold">{order.product_id}</span>
            <span class={`px-2 py-1 rounded text-sm ${
              order.side === "BUY" ? "bg-success text-white" : "bg-danger text-white"
            }`}>
              {order.side}
            </span>
          </div>
          <div class="text-sm text-gray-600">
            <div class="flex justify-between">
              <span>Type:</span>
              <span>{order.order_type}</span>
            </div>
            <div class="flex justify-between">
              <span>Size:</span>
              <span>{order.base_size}</span>
            </div>
            {order.limit_price && (
              <div class="flex justify-between">
                <span>Price:</span>
                <span>${order.limit_price}</span>
              </div>
            )}
            <div class="flex justify-between">
              <span>Status:</span>
              <span>{order.status}</span>
            </div>
            <div class="flex justify-between">
              <span>Filled:</span>
              <span>{order.filled_size}</span>
            </div>
            <div class="text-xs text-gray-500 mt-2">
              {new Date(order.created_time).toLocaleString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChartComponent({ 
  pair, 
  timeframe,
  chartRef,
  candleSeriesRef: _candleSeriesRef,
  onError
}: { 
  pair: string; 
  timeframe: string;
  chartRef: any;
  candleSeriesRef: any;
  onError?: (message: string) => void;
}) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const [indicators, setIndicators] = useState<IndicatorSettings>({
    keyLevels: true,
    movingAverages: true,
    rsi: true,
    volume: true,
    macd: false,
    bollinger: false,
    ichimoku: false
  });

  const toggleIndicator = (name: keyof IndicatorSettings) => {
    setIndicators(prev => ({ ...prev, [name]: !prev[name] }));
  };

  // Initialize chart on client-side only
  useEffect(() => {
    let echartsModule: any;
    
    const initChart = async () => {
      try {
        // Dynamically import echarts only on client side
        echartsModule = await import("npm:echarts@5.4.3");
        const echarts = echartsModule.default || echartsModule;
        
        if (!chartContainerRef.current) return;

        // Initialize ECharts instance
        chartInstanceRef.current = echarts.init(chartContainerRef.current);
        chartRef.current = chartInstanceRef.current;

    const updateData = async (currentPair: string) => {
      try {
        const response = await fetch(`${API_BASE_URL}/market_data/${currentPair}?timeframe=${timeframe}`);
        if (!response.ok) throw new Error('Failed to fetch candle data');
        
        const data = await response.json();
        if (!data.candles || !Array.isArray(data.candles)) {
          throw new Error('Invalid candle data format');
        }

        const reversedCandles = [...data.candles].reverse();
        const categoryData = reversedCandles.map((d: Candle) => {
          const date = new Date(d.start * 1000);
          return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          });
        });

        const candlestickData = reversedCandles.map((d: Candle) => [
          Number(d.open),
          Number(d.close),
          Number(d.low),
          Number(d.high)
        ]);

        const closes = reversedCandles.map((d: Candle) => Number(d.close));
        const ma20Data = calculateMA(closes, 20);
        const ma50Data = calculateMA(closes, 50);
        const rsiData = calculateRSI(closes);

        // Set chart options
        const options = {
          animation: false,
          legend: {
            data: [
              'Candlestick', 'MA20', 'MA50', 'Volume', 'RSI',
              'BB Middle', 'BB Upper', 'BB Lower',
              'MACD', 'Signal', 'Histogram',
              'Conversion', 'Base', 'Span A', 'Span B'
            ],
            selected: {
              'MA20': indicators.movingAverages,
              'MA50': indicators.movingAverages,
              'Volume': indicators.volume,
              'RSI': indicators.rsi,
              'BB Middle': indicators.bollinger,
              'BB Upper': indicators.bollinger,
              'BB Lower': indicators.bollinger,
              'MACD': indicators.macd,
              'Signal': indicators.macd,
              'Histogram': indicators.macd,
              'Conversion': indicators.ichimoku,
              'Base': indicators.ichimoku,
              'Span A': indicators.ichimoku,
              'Span B': indicators.ichimoku
            }
          },
          tooltip: {
            trigger: 'axis',
            axisPointer: {
              type: 'cross'
            },
            formatter: (params: any[]) => {
              if (!params || params.length === 0) return '';
              
              // Format the date
              const date = new Date(params[0].axisValue);
              const timeStr = date.toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              });

              // Build tooltip content
              let tooltipContent = `<div style="font-weight:bold">${timeStr}</div>`;
              
              // Add each series data
              params.forEach(param => {
                const color = param.color || param.borderColor;
                const marker = `<span style="display:inline-block;margin-right:4px;border-radius:50%;width:10px;height:10px;background-color:${color};"></span>`;
                
                let value = param.value;
                if (param.seriesName === 'Candlestick') {
                  // For candlestick, show OHLC values
                  const [open, close, low, high] = param.data;
                  tooltipContent += `<div style="margin:4px 0">
                    ${marker}${param.seriesName}<br/>
                    Open: ${open}<br/>
                    High: ${high}<br/>
                    Low: ${low}<br/>
                    Close: ${close}
                  </div>`;
                } else {
                  // For other series (MA, RSI, Volume)
                  tooltipContent += `<div style="margin:4px 0">
                    ${marker}${param.seriesName}: ${typeof value === 'number' ? value.toFixed(2) : value}
                  </div>`;
                }
              });

              return tooltipContent;
            }
          },
          grid: [{
            left: '10%',
            right: '10%',
            height: '50%'
          }, {
            left: '10%',
            right: '10%',
            top: '63%',
            height: '15%'
          }, {
            left: '10%',
            right: '10%',
            top: '81%',
            height: '15%'
          }],
          xAxis: [{
            type: 'category',
            data: categoryData,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false }
          }, {
            type: 'category',
            gridIndex: 1,
            data: categoryData,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false }
          }, {
            type: 'category',
            gridIndex: 2,
            data: categoryData,
            boundaryGap: false,
            axisLine: { onZero: false },
            splitLine: { show: false }
          }],
          yAxis: [{
            scale: true,
            splitArea: {
              show: true
            }
          }, {
            gridIndex: 1,
            splitNumber: 3,
            axisLine: { show: false },
            axisLabel: { show: false },
            splitLine: { show: false }
          }, {
            gridIndex: 2,
            scale: true,
            splitNumber: 4,
            axisLabel: { show: true },
            axisLine: { show: false },
            splitLine: { show: true }
          }],
          dataZoom: [{
            type: 'inside',
            xAxisIndex: [0, 1, 2],
            start: 75,
            end: 100
          }, {
            show: true,
            xAxisIndex: [0, 1, 2],
            type: 'slider',
            bottom: '0%',
            start: 75,
            end: 100
          }],
          series: [
            {
        name: 'Candlestick',
        type: 'candlestick',
        data: candlestickData,
        itemStyle: {
          color: '#26a69a',      // Bullish candles (close >= open)
          color0: '#ef5350',     // Bearish candles (close < open)
          borderColor: '#26a69a', // Bullish border
          borderColor0: '#ef5350',// Bearish border
          borderWidth: 2
        }
      },
      {
        name: 'Volume',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: reversedCandles.map((d: Candle, i: number) => [
          i,
          Number(d.volume),
          Number(d.close) > Number(d.open) ? 1 : -1
        ]),
        itemStyle: {
          color: (params: any) => {
            return params.data[2] > 0 ? 'rgba(38, 166, 154, 0.3)' : 'rgba(239, 83, 80, 0.3)';
          }
        }
      },
      ...(indicators.movingAverages ? [
        {
          name: 'MA20',
          type: 'line',
          data: ma20Data.map((value, i) => [
            i,  // Use index for category axis
            value
          ]),
          smooth: true,
          lineStyle: {
            opacity: 0.8,
            color: 'rgba(255, 192, 0, 0.8)'
          },
          showSymbol: false
        },
        {
          name: 'MA50',
          type: 'line',
          data: ma50Data.map((value, i) => [
            i,  // Use index for category axis
            value
          ]),
          smooth: true,
          lineStyle: {
            opacity: 0.8,
            color: 'rgba(0, 150, 255, 0.8)'
          },
          showSymbol: false
        }
      ] : []),
      ...(indicators.rsi ? [{
        name: 'RSI',
        type: 'line',
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: rsiData.map((value, i) => [
          i,  // Use index for category axis
          value
        ]),
        lineStyle: {
          color: '#2962FF',
          width: 1
        },
        showSymbol: false,
        markLine: {
          silent: true,
          symbol: ['none', 'none'],
          label: { show: true },
          data: [
            {
              yAxis: 70,
              lineStyle: { color: '#ef5350', type: 'dashed' },
              label: { formatter: 'Overbought (70)' }
            },
            {
              yAxis: 30,
              lineStyle: { color: '#26a69a', type: 'dashed' },
              label: { formatter: 'Oversold (30)' }
            }
          ]
        }
      }] : []),
      ...(indicators.bollinger ? (() => {
        const closes = reversedCandles.map((d: Candle) => Number(d.close));
        const bands = calculateBollingerBands(closes);
        return [{
          name: 'BB Middle',
          type: 'line',
          data: bands.middle.map((value, i) => [i, value]),
          lineStyle: { opacity: 0.5, color: '#9ca3af' },
          showSymbol: false
        }, {
          name: 'BB Upper',
          type: 'line',
          data: bands.upper.map((value, i) => [i, value]),
          lineStyle: { opacity: 0.5, color: '#9ca3af', type: 'dashed' },
          showSymbol: false
        }, {
          name: 'BB Lower',
          type: 'line',
          data: bands.lower.map((value, i) => [i, value]),
          lineStyle: { opacity: 0.5, color: '#9ca3af', type: 'dashed' },
          showSymbol: false
        }];
      })() : []),
      ...(indicators.macd ? (() => {
        const closes = reversedCandles.map((d: Candle) => Number(d.close));
        const macdData = calculateMACD(closes);
        return [{
          name: 'MACD',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: macdData.macd.map((value, i) => [i, value]),
          lineStyle: { color: '#2962FF' },
          showSymbol: false
        }, {
          name: 'Signal',
          type: 'line',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: macdData.signal.map((value, i) => [i, value]),
          lineStyle: { color: '#FF6B6B' },
          showSymbol: false
        }, {
          name: 'Histogram',
          type: 'bar',
          xAxisIndex: 2,
          yAxisIndex: 2,
          data: macdData.histogram.map((value, i) => [i, value]),
          itemStyle: {
            color: (params: any) => params.data[1] >= 0 ? '#26a69a' : '#ef5350'
          }
        }];
      })() : []),
      ...(indicators.ichimoku ? (() => {
        const ichimokuData = calculateIchimoku({
          high: reversedCandles.map((d: Candle) => Number(d.high)),
          low: reversedCandles.map((d: Candle) => Number(d.low)),
          close: reversedCandles.map((d: Candle) => Number(d.close))
        });
        return [{
          name: 'Conversion',
          type: 'line',
          data: ichimokuData.conversion.map((value, i) => [i, value]),
          lineStyle: { color: '#2962FF' },
          showSymbol: false
        }, {
          name: 'Base',
          type: 'line',
          data: ichimokuData.base.map((value, i) => [i, value]),
          lineStyle: { color: '#FF6B6B' },
          showSymbol: false
        }, {
          name: 'Span A',
          type: 'line',
          data: ichimokuData.spanA.map((value, i) => [i, value]),
          lineStyle: { color: '#26a69a' },
          showSymbol: false,
          areaStyle: {
            color: '#26a69a',
            opacity: 0.1
          }
        }, {
          name: 'Span B',
          type: 'line',
          data: ichimokuData.spanB.map((value, i) => [i, value]),
          lineStyle: { color: '#ef5350' },
          showSymbol: false,
          areaStyle: {
            color: '#ef5350',
            opacity: 0.1
          }
        }];
      })() : [])
    ]

        };

        chartInstanceRef.current.setOption(options, true);
      } catch (err) {
        console.error('Error updating chart:', err);
        if (onError) onError(err instanceof Error ? err.message : 'Failed to update chart');
      }
    };

    // Handle window resize
    const handleResize = () => {
          if (chartInstanceRef.current) {
            chartInstanceRef.current.resize();
          }
        };

        globalThis.addEventListener('resize', handleResize);
        await updateData(pair);

    return () => {
          globalThis.removeEventListener('resize', handleResize);
          if (chartInstanceRef.current) {
            chartInstanceRef.current.dispose();
          }
        };
      } catch (err) {
        console.error('Error initializing chart:', err);
        if (onError) onError(err instanceof Error ? err.message : 'Failed to initialize chart');
      }
    };

    // Only run on client-side
    if (typeof document !== 'undefined') {
      initChart();
    }
  }, [pair, timeframe, indicators]);

  return (
    <div class="relative bg-white rounded-lg shadow-lg overflow-hidden">
      <div class="flex gap-2 p-2 bg-gray-100">
        <button
          onClick={() => toggleIndicator('keyLevels')}
          class={`px-3 py-1 rounded ${indicators.keyLevels ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        >
          Key Levels
        </button>
        <button
          onClick={() => toggleIndicator('movingAverages')}
          class={`px-3 py-1 rounded ${indicators.movingAverages ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        >
          Moving Averages
        </button>
        <button
          onClick={() => toggleIndicator('bollinger')}
          class={`px-3 py-1 rounded ${indicators.bollinger ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        >
          Bollinger Bands
        </button>
        <button
          onClick={() => toggleIndicator('ichimoku')}
          class={`px-3 py-1 rounded ${indicators.ichimoku ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        >
          Ichimoku Cloud
        </button>
        <button
          onClick={() => toggleIndicator('macd')}
          class={`px-3 py-1 rounded ${indicators.macd ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
        >
          MACD
        </button>
      </div>
      <div ref={chartContainerRef} class="w-full" style="height: 600px"></div>
    </div>
  );
}

// Helper functions for calculating indicators
function calculateMA(data: number[], period: number): number[] {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(NaN);
      continue;
    }
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += data[i - j];
    }
    result.push(sum / period);
  }
  return result;
}

function calculateRSI(data: number[], period: number = 14): number[] {
  const result: number[] = [];
  const changes: number[] = [];
  
  // Calculate price changes
  for (let i = 1; i < data.length; i++) {
    changes.push(data[i] - data[i - 1]);
  }

  // Add initial NaN values for the first period
  for (let i = 0; i < period; i++) {
    result.push(NaN);
  }

  // Calculate first average gain and loss
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 0; i < period; i++) {
    const change = changes[i];
    if (change > 0) avgGain += change;
    if (change < 0) avgLoss += Math.abs(change);
  }
  avgGain = avgGain / period;
  avgLoss = avgLoss / period;

  // Calculate RSI using Wilder's smoothing
  result.push(100 - (100 / (1 + (avgGain / avgLoss))));

  // Calculate subsequent values
  for (let i = period + 1; i < data.length; i++) {
    const change = changes[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = ((avgGain * (period - 1)) + gain) / period;
    avgLoss = ((avgLoss * (period - 1)) + loss) / period;

    const rs = avgGain / avgLoss;
    result.push(100 - (100 / (1 + rs)));
  }

  return result;
}

function calculateBollingerBands(data: number[], period: number = 20, stdDev: number = 2): { 
  middle: number[];
  upper: number[];
  lower: number[];
} {
  const middle = calculateMA(data, period);
  const upper: number[] = [];
  const lower: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      upper.push(NaN);
      lower.push(NaN);
      continue;
    }

    const slice = data.slice(i - period + 1, i + 1);
    const std = Math.sqrt(
      slice.reduce((sum, val) => sum + Math.pow(val - middle[i], 2), 0) / period
    );

    upper.push(middle[i] + stdDev * std);
    lower.push(middle[i] - stdDev * std);
  }

  return { middle, upper, lower };
}

function calculateMACD(data: number[]): {
  macd: number[];
  signal: number[];
  histogram: number[];
} {
  const fast = calculateMA(data, 12);
  const slow = calculateMA(data, 26);
  const macd: number[] = [];
  
  for (let i = 0; i < data.length; i++) {
    macd.push(fast[i] - slow[i]);
  }
  
  const signal = calculateMA(macd, 9);
  const histogram = macd.map((value, i) => value - signal[i]);
  
  return { macd, signal, histogram };
}

function calculateIchimoku(data: { high: number[]; low: number[]; close: number[] }): {
  conversion: number[];
  base: number[];
  spanA: number[];
  spanB: number[];
  laggingSpan: number[];
} {
  const conversion: number[] = [];
  const base: number[] = [];
  const spanA: number[] = [];
  const spanB: number[] = [];
  const laggingSpan = data.close.map((val, i) => i >= 26 ? val : NaN);

  for (let i = 0; i < data.high.length; i++) {
    // Conversion Line (9)
    if (i >= 8) {
      const highSlice = data.high.slice(i - 8, i + 1);
      const lowSlice = data.low.slice(i - 8, i + 1);
      conversion.push((Math.max(...highSlice) + Math.min(...lowSlice)) / 2);
    } else {
      conversion.push(NaN);
    }

    // Base Line (26)
    if (i >= 25) {
      const highSlice = data.high.slice(i - 25, i + 1);
      const lowSlice = data.low.slice(i - 25, i + 1);
      base.push((Math.max(...highSlice) + Math.min(...lowSlice)) / 2);
    } else {
      base.push(NaN);
    }
  }

  // Span A and B
  for (let i = 0; i < data.high.length; i++) {
    if (i >= 25) {
      spanA.push((conversion[i] + base[i]) / 2);
    } else {
      spanA.push(NaN);
    }

    if (i >= 51) {
      const highSlice = data.high.slice(i - 51, i + 1);
      const lowSlice = data.low.slice(i - 51, i + 1);
      spanB.push((Math.max(...highSlice) + Math.min(...lowSlice)) / 2);
    } else {
      spanB.push(NaN);
    }
  }

  return { conversion, base, spanA, spanB, laggingSpan };
}

function formatIndicatorName(name: string) {
  return name
    .split("_")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

function formatIndicatorValue(value: any) {
  if (typeof value === "number") {
    return value.toFixed(2);
  }
  if (Array.isArray(value)) {
    return value.map(v => v.toString()).join(", ");
  }
  
  // If it's an object (like a dictionary), format it as a proper JSON display
  if (typeof value === "object" && value !== null) {
    try {
      return (
        <pre className="text-sm overflow-auto max-h-60 bg-gray-100 p-2 rounded">
          {JSON.stringify(value, null, 2)}
        </pre>
      );
    } catch (e) {
      return value.toString();
    }
  }
  
  return value.toString();
}

export default function Chart({ initialPair = "BTC-USD" }: ChartProps) {
  const pair = useSignal(initialPair);
  const timeframe = useSignal("1h");
  const loading = useSignal(false);
  const error = useSignal("");
  const orders = useSignal<Order[]>([]);
  const analysisResult = useSignal<AnalysisResult | null>(null);
  const orderType = useSignal<"MARKET" | "LIMIT">("MARKET");
  const limitPrice = useSignal<string>("");
  const chartRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const amount = useSignal("1000");
  const availablePairs = useSignal<string[]>([]);
  const showOrders = useSignal(false);
  const portfolioData = useSignal<any>(null);
  const showDetailedAnalysis = useSignal(false);
  const priceLineRefs = useRef<any[]>([]);

  useEffect(() => {
    const initializeClientSide = async () => {
      try {
        // Load particles.js
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js";
    script.async = true;
    script.onload = initParticles;
    document.head.appendChild(script);

        // Fetch initial data
        await Promise.all([
          fetchPairs(),
          fetchOrders(),
          fetchPortfolio()
        ]);

    return () => {
          if (script.parentNode) {
            script.parentNode.removeChild(script);
          }
        };
      } catch (err) {
        console.error('Error initializing:', err);
      }
    };

    // Only run on client-side
    if (typeof document !== 'undefined') {
      initializeClientSide();
    }
  }, []);

  const initParticles = () => {
    // @ts-ignore - particles.js is loaded from CDN
    particlesJS("particles-js", {
      particles: {
        number: { value: 80, density: { enable: true, value_area: 800 } },
        color: { value: "#3b82f6" },
        shape: { type: "circle" },
        opacity: { value: 0.5, random: false },
        size: { value: 3, random: true },
        line_linked: {
          enable: true,
          distance: 150,
          color: "#3b82f6",
          opacity: 0.4,
          width: 1,
        },
        move: {
          enable: true,
          speed: 6,
          direction: "none",
          random: false,
          straight: false,
          out_mode: "out",
          bounce: false,
        },
      },
      interactivity: {
        detect_on: "canvas",
        events: {
          onhover: { enable: true, mode: "repulse" },
          onclick: { enable: true, mode: "push" },
          resize: true,
        },
      },
      retina_detect: true,
    });
  };

  const fetchPortfolio = async () => {
    try {
      const response = await fetch('http://localhost:8001/portfolio');
      const data = await response.json();
      if (data.portfolio) {
        portfolioData.value = data.portfolio;
      }
    } catch (err) {
      console.error('Error fetching portfolio:', err);
    }
  };

  const fetchPairs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/pairs`);
      const data = await response.json();
      if (data.pairs) {
        availablePairs.value = data.pairs;
      }
    } catch (err) {
      console.error('Error fetching pairs:', err);
    }
  };

  const fetchOrders = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/orders`);
      const data = await response.json();
      if (data.orders) {
        orders.value = data.orders;
      }
    } catch (err) {
      console.error('Error fetching orders:', err);
    }
  };

  const handleAnalyze = async () => {
    if (!pair.value || !timeframe.value || !amount.value) {
      error.value = "Please fill in all fields";
      return;
    }

    loading.value = true;
    error.value = "";

    try {
      const requestBody = {
        pair: pair.value,
        timeframe: timeframe.value,
        amount: parseFloat(amount.value),
        strategy: "default",
        balances: [
          { currency: 'XRP', balance: 2.694162 },
          { currency: 'USD', balance: 104.119292726666 }
        ]
      };

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Failed with status ${response.status}.\n`;
        
        try {
          const errorData = JSON.parse(errorText);
          errorMessage += errorData.error || errorText;
        } catch {
          errorMessage += errorText;
        }
        
        error.value = `Analysis Error: ${errorMessage}`;
        throw new Error(errorMessage);
      }

      const data = await response.json();

      // Validate required fields
      const requiredFields = [
        'signal', 'trade_size', 'take_profit', 'stop_loss', 
        'confidence', 'explanation', 'key_levels', 'analysis_details'
      ];

      const missingFields = requiredFields.filter(field => !data[field]);
      if (missingFields.length > 0) {
        error.value = `Response missing required fields: ${missingFields.join(', ')}`;
        throw new Error(error.value);
      }

      error.value = ""; // Clear error if successful

      // Update the analysis result with validated data
      analysisResult.value = {
        ...data,
        pair: pair.value,
        timeframe: timeframe.value,
        key_levels: {
          support_levels: Array.isArray(data.key_levels.support_levels) ? data.key_levels.support_levels : [data.key_levels.support_levels],
          resistance_levels: Array.isArray(data.key_levels.resistance_levels) ? data.key_levels.resistance_levels : [data.key_levels.resistance_levels],
          targets: Array.isArray(data.key_levels.targets) ? data.key_levels.targets : [data.key_levels.targets]
        }
      };

      // Update chart markers for key levels if we have valid data
      if (analysisResult.value && chartRef.current && candleSeriesRef.current) {
        // Clear existing markers
        candleSeriesRef.current.setMarkers([]);
        
        // Remove existing price lines
        priceLineRefs.current.forEach(line => {
          if (line && candleSeriesRef.current) {
            candleSeriesRef.current.removePriceLine(line);
          }
        });
        priceLineRefs.current = []; // Clear the refs array
        
        const currentTime = Math.floor(Date.now() / 1000);
        
        // Add support levels as price lines
        analysisResult.value.key_levels.support_levels.forEach(level => {
          const line = candleSeriesRef.current.createPriceLine({
            price: level.price,
            color: '#10b981',
            lineWidth: 2,
            lineStyle: 2, // Dashed
            axisLabelVisible: true,
            title: `Support: $${level.price.toLocaleString()}`
          });
          priceLineRefs.current.push(line);
        });

        // Add resistance levels as price lines
        analysisResult.value.key_levels.resistance_levels.forEach(level => {
          const line = candleSeriesRef.current.createPriceLine({
            price: level.price,
            color: '#ef4444',
            lineWidth: 2,
            lineStyle: 2, // Dashed
            axisLabelVisible: true,
            title: `Resistance: $${level.price.toLocaleString()}`
          });
          priceLineRefs.current.push(line);
        });

        // Add take profit lines
        const tp1Value = parseFloat(analysisResult.value.take_profit.tp1.split(' ')[0]);
        const tp2Value = parseFloat(analysisResult.value.take_profit.tp2.split(' ')[0]);
        const tp3Value = parseFloat(analysisResult.value.take_profit.tp3.split(' ')[0]);

        [
          { price: tp1Value, title: `TP1: ${analysisResult.value.take_profit.tp1}` },
          { price: tp2Value, title: `TP2: ${analysisResult.value.take_profit.tp2}` },
          { price: tp3Value, title: `TP3: ${analysisResult.value.take_profit.tp3}` }
        ].forEach((tp) => {
          if (!isNaN(tp.price)) {
            const tpLine = candleSeriesRef.current.createPriceLine({
              price: tp.price,
              color: '#22c55e',
              lineWidth: 2,
              lineStyle: 1, // Solid
              axisLabelVisible: true,
              title: tp.title
            });
            priceLineRefs.current.push(tpLine);
          }
        });

        // Add stop loss lines
        const slInitialValue = parseFloat(analysisResult.value.stop_loss.initial.split(' ')[0]);
        const slBreakevenValue = parseFloat(analysisResult.value.stop_loss.breakeven.split(' ')[0]);

        [
          { price: slInitialValue, title: `Initial SL: ${analysisResult.value.stop_loss.initial}`, color: '#ef4444' },
          { price: slBreakevenValue, title: `Breakeven: ${analysisResult.value.stop_loss.breakeven}`, color: '#f97316' }
        ].forEach((sl) => {
          if (!isNaN(sl.price)) {
            const slLine = candleSeriesRef.current.createPriceLine({
              price: sl.price,
              color: sl.color,
              lineWidth: 2,
              lineStyle: 1, // Solid
              axisLabelVisible: true,
              title: sl.title
            });
            priceLineRefs.current.push(slLine);
          }
        });

        // Add target levels as markers
        const targetMarkers = analysisResult.value.key_levels.targets.map(level => ({
          time: currentTime,
          position: 'inBar' as const,
          color: '#3b82f6',
          shape: 'circle' as const,
          text: `Target: $${level.toFixed(2)}`,
          size: 1
        }));

        // Set target markers
        candleSeriesRef.current.setMarkers(targetMarkers);
      }

    } catch (err: unknown) {
      console.error("Analysis error:", err);
      error.value = err instanceof Error ? err.message : "Failed to analyze";
      analysisResult.value = null;
    } finally {
      loading.value = false;
    }
  };

  const handleTrade = async () => {
    if (!analysisResult.value) return;

    try {
      loading.value = true;
      error.value = "";

      // Validate limit price for limit orders
      if (orderType.value === "LIMIT" && (!limitPrice.value || parseFloat(limitPrice.value) <= 0)) {
        throw new Error("Please enter a valid limit price");
      }

      // Format the trade size to ensure it's a valid decimal string
      const tradeSize = parseFloat(analysisResult.value.trade_size).toFixed(8);

      // Extract the numeric value from take profit and stop loss
      const takeProfitValue = parseFloat(analysisResult.value.take_profit.tp1.split(' ')[0]);
      const stopLossValue = parseFloat(analysisResult.value.stop_loss.initial.split(' ')[0]);

      const tradeData = {
        product_id: analysisResult.value.pair,
        side: analysisResult.value.signal,
        order_type: orderType.value,
        base_size: tradeSize,
        limit_price: orderType.value === "LIMIT" ? parseFloat(limitPrice.value).toFixed(8) : undefined,
        take_profit: takeProfitValue,
        stop_loss: stopLossValue
      };

      console.log("Executing trade with data:", tradeData);

      const response = await fetch(`${API_BASE_URL}/execute-trade`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(tradeData),
      });

      const responseData = await response.json();

      if (!response.ok) {
        throw new Error(responseData.detail || 'Failed to execute trade');
      }
      
      // Refresh orders list
      await fetchOrders();
      
      // Show success message with order details
      error.value = `Trade executed successfully! Order ID: ${responseData.order.id}
        Type: ${responseData.order.order_type}
        Size: ${responseData.order.base_size}
        ${responseData.order.limit_price ? `Price: $${responseData.order.limit_price}` : ''}
        Take Profit: $${takeProfitValue.toFixed(8)}
        Stop Loss: $${stopLossValue.toFixed(8)}`;
      
      // Reset form
      orderType.value = "MARKET";
      limitPrice.value = "";
      
    } catch (err) {
      console.error("Trade execution error:", err);
      error.value = err instanceof Error ? err.message : 'Failed to execute trade';
    } finally {
      loading.value = false;
    }
  };

  return (
    <div class="min-h-screen bg-[#f8fafc]">
      <div id="particles-js" class="fixed w-full h-full top-0 left-0 -z-10 opacity-10" />
      
      <div class="container mx-auto px-4 py-8">
        <div class="bg-white rounded-lg shadow-lg p-6">
          <h1 class="text-3xl font-bold text-center text-primary mb-8 font-montserrat">
            TradeGPT
            <span class="block text-sm font-normal text-neutral mt-2">
              AI-Powered Crypto Trading Analysis
            </span>
          </h1>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="input-group">
              <label class="block text-sm font-semibold mb-2" for="pair">
                Trading Pair
              </label>
              <select
                id="pair"
                class="w-full p-3 border-2 border-gray-200 rounded-lg focus:border-accent focus:outline-none transition-colors"
                value={pair.value}
                onChange={(e) => pair.value = (e.target as HTMLSelectElement).value}
              >
                {availablePairs.value.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            <div class="input-group">
              <label class="block text-sm font-semibold mb-2" for="timeframe">
                Timeframe
              </label>
              <select
                id="timeframe"
                class="w-full p-3 border-2 border-gray-200 rounded-lg focus:border-accent focus:outline-none transition-colors"
                value={timeframe.value}
                onChange={(e) => timeframe.value = (e.target as HTMLSelectElement).value}
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1d">1 Day</option>
              </select>
            </div>

            <div class="input-group">
              <label class="block text-sm font-semibold mb-2" for="amount">
                Amount (USD)
              </label>
              <input
                type="number"
                id="amount"
                class="w-full p-3 border-2 border-gray-200 rounded-lg focus:border-accent focus:outline-none transition-colors"
                value={amount.value}
                onChange={(e) => amount.value = (e.target as HTMLInputElement).value}
                min="0"
                step="100"
              />
            </div>
          </div>

          <ChartComponent 
            pair={pair.value} 
            timeframe={timeframe.value} 
            chartRef={chartRef}
            candleSeriesRef={candleSeriesRef}
            onError={(message) => error.value = message}
            data-chart 
          />

          <div class="flex gap-4 mb-8">
            <button
              onClick={handleAnalyze}
              disabled={loading.value}
              class="flex-1 bg-accent text-white font-semibold py-3 px-6 rounded-lg hover:bg-accent-hover transition-colors disabled:bg-neutral disabled:cursor-not-allowed"
            >
              {loading.value ? (
                <span class="flex items-center justify-center">
                  <i class="fas fa-spinner fa-spin mr-2"></i>
                  Analyzing...
                </span>
              ) : (
                "Analyze"
              )}
            </button>

            {analysisResult.value && (
              <button
                onClick={() => showDetailedAnalysis.value = !showDetailedAnalysis.value}
                class="bg-primary text-white font-semibold py-3 px-6 rounded-lg hover:bg-primary-dark transition-colors"
              >
                <i class={`fas fa-chart-line mr-2`}></i>
                {showDetailedAnalysis.value ? "Hide Analysis" : "Show Analysis"}
              </button>
            )}

            <button
              onClick={() => showOrders.value = !showOrders.value}
              class="bg-primary text-white font-semibold py-3 px-6 rounded-lg hover:bg-primary-dark transition-colors"
            >
              {showOrders.value ? "Hide Orders" : "Show Orders"}
            </button>
          </div>

          {error.value && (
            <div class={`mt-6 p-4 rounded-lg text-center ${
              error.value.startsWith("Sending") || error.value.startsWith("Received")
                ? "bg-blue-100 text-blue-700"
                : "bg-red-100 text-danger"
            }`}>
              <i class={`fas ${
                error.value.startsWith("Sending") || error.value.startsWith("Received")
                  ? "fa-info-circle"
                  : "fa-exclamation-circle"
              } mr-2`}></i>
              <pre class="whitespace-pre-wrap font-mono text-sm">{error.value}</pre>
            </div>
          )}

          {/* Quick Analysis Summary - Always visible when there's a result */}
          {analysisResult.value && (
            <div class="bg-white rounded-lg shadow-lg p-6 mt-8">
              <div class="flex justify-between items-center mb-6 pb-4 border-b">
                <div>
                  <div class="text-2xl font-bold text-primary">{analysisResult.value.pair}</div>
                  <div class="text-xl text-neutral mt-2">
                    Current Price: ${analysisResult.value.analysis_details.current_price.toFixed(2)}
                  </div>
                </div>
                <div class={`px-4 py-2 rounded-lg font-semibold ${
                  analysisResult.value.signal === "BUY" ? "bg-success text-white" : "bg-danger text-white"
                }`}>
                  {analysisResult.value.signal}
                </div>
              </div>

              {/* Main Analysis Grid - Quick Summary */}
              <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div class="bg-secondary p-4 rounded-lg">
                  <div class="text-sm text-neutral mb-1">Trade Size</div>
                  <div class="text-lg font-semibold flex items-center justify-between">
                    <span>{analysisResult.value.trade_size}</span>
                    <i class="fas fa-coins text-accent"></i>
                  </div>
                </div>
                <div class="bg-secondary p-4 rounded-lg">
                  <div class="text-sm text-neutral mb-1">Take Profit</div>
                  <div class="space-y-2">
                    <div class="text-sm font-semibold flex items-center justify-between">
                      <span>TP1</span>
                      <span class="text-success">${analysisResult.value.take_profit.tp1}</span>
                    </div>
                    <div class="text-sm font-semibold flex items-center justify-between">
                      <span>TP2</span>
                      <span class="text-success">${analysisResult.value.take_profit.tp2}</span>
                    </div>
                    <div class="text-sm font-semibold flex items-center justify-between">
                      <span>TP3</span>
                      <span class="text-success">${analysisResult.value.take_profit.tp3}</span>
                    </div>
                  </div>
                </div>
                <div class="bg-secondary p-4 rounded-lg">
                  <div class="text-sm text-neutral mb-1">Stop Loss</div>
                  <div class="space-y-2">
                    <div class="text-sm font-semibold flex items-center justify-between">
                      <span>Initial</span>
                      <span class="text-danger">${analysisResult.value.stop_loss.initial}</span>
                    </div>
                    <div class="text-sm font-semibold flex items-center justify-between">
                      <span>Breakeven</span>
                      <span class="text-warning">${analysisResult.value.stop_loss.breakeven}</span>
                    </div>
                    <div class="text-sm font-semibold flex items-center justify-between">
                      <span>Trailing</span>
                      <span class="text-warning">${analysisResult.value.stop_loss.trailing}</span>
                    </div>
                  </div>
                </div>
                <div class="bg-secondary p-4 rounded-lg">
                  <div class="text-sm text-neutral mb-1">Support and Resistance</div>
                  <div class="space-y-2">
                    <div>
                      <div class="text-sm font-semibold mb-1">Support Levels:</div>
                      {analysisResult.value.key_levels.support_levels.map((level, index) => (
                        <div key={index} class="flex items-center justify-between text-sm">
                          <span>${level.price.toLocaleString()}</span>
                          <span class="text-neutral capitalize">
                            ({level.strength} - {level.type})
                          </span>
                        </div>
                      ))}
                    </div>
                    <div class="mt-2">
                      <div class="text-sm font-semibold mb-1">Resistance Levels:</div>
                      {analysisResult.value.key_levels.resistance_levels.map((level, index) => (
                        <div key={index} class="flex items-center justify-between text-sm">
                          <span>${level.price.toLocaleString()}</span>
                          <span class="text-neutral capitalize">
                            ({level.strength} - {level.type})
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div class="bg-secondary p-4 rounded-lg">
                  <div class="text-sm text-neutral mb-1">Confidence</div>
                  <div class="text-lg font-semibold flex items-center justify-between">
                    <span class={`${
                      analysisResult.value.confidence === "HIGH" ? "text-success" :
                      analysisResult.value.confidence === "MEDIUM" ? "text-warning" :
                      "text-danger"
                    }`}>
                      {analysisResult.value.confidence}
                    </span>
                    <i class={`fas fa-${
                      analysisResult.value.confidence === "HIGH" ? "check-circle text-success" :
                      analysisResult.value.confidence === "MEDIUM" ? "exclamation-circle text-warning" :
                      "times-circle text-danger"
                    }`}></i>
                  </div>
                </div>
              </div>

              {/* Trade Execution Panel */}
              <div class="mt-8 border-t pt-8">
                <h2 class="text-xl font-semibold text-primary mb-4">Execute Trade</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label class="block text-sm font-semibold mb-2">Order Type</label>
                    <select
                      class="w-full p-3 border-2 border-gray-200 rounded-lg focus:border-accent focus:outline-none transition-colors"
                      value={orderType.value}
                      onChange={(e) => orderType.value = (e.target as HTMLSelectElement).value as "MARKET" | "LIMIT"}
                    >
                      <option value="MARKET">Market</option>
                      <option value="LIMIT">Limit</option>
                    </select>
                  </div>

                  {orderType.value === "LIMIT" && (
                    <div>
                      <label class="block text-sm font-semibold mb-2">Limit Price</label>
                      <input
                        type="number"
                        class="w-full p-3 border-2 border-gray-200 rounded-lg focus:border-accent focus:outline-none transition-colors"
                        value={limitPrice.value}
                        onChange={(e) => limitPrice.value = (e.target as HTMLInputElement).value}
                        step="0.00000001"
                        min="0"
                      />
                    </div>
                  )}

                  <div class="md:col-span-2">
                    <button
                      onClick={handleTrade}
                      class={`w-full py-3 px-6 rounded-lg font-semibold text-white transition-colors ${
                        analysisResult.value.signal === "BUY"
                          ? "bg-success hover:bg-success/80"
                          : "bg-danger hover:bg-danger/80"
                      }`}
                      disabled={loading.value}
                    >
                      {loading.value ? (
                        <span class="flex items-center justify-center">
                          <i class="fas fa-spinner fa-spin mr-2"></i>
                          Executing Trade...
                        </span>
                      ) : (
                        `Execute ${analysisResult.value.signal} ${orderType.value} Order`
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Detailed Analysis Section - Collapsible */}
          {showDetailedAnalysis.value && analysisResult.value && (
            <div class="bg-white rounded-lg shadow-lg p-6 mt-8 animate-fade-in">
              {/* Confidence and Explanation */}
              <div class="mb-8">
                <div class="text-lg font-semibold mb-2">Analysis Details</div>
                <div class="relative w-full h-2 bg-gray-200 rounded mb-2">
                  <div
                    class={`absolute top-0 left-0 h-full rounded ${
                      analysisResult.value.confidence === "HIGH" ? "bg-success" :
                      analysisResult.value.confidence === "MEDIUM" ? "bg-warning" :
                      "bg-danger"
                    }`}
                    style={{ width: `${
                      analysisResult.value.confidence === "HIGH" ? "100" :
                      analysisResult.value.confidence === "MEDIUM" ? "66" :
                      "33"
                    }%` }}
                  />
                </div>
                <div class="mt-4 p-4 bg-secondary rounded-lg">
                  <div class="text-sm text-neutral mb-1">Analysis Explanation</div>
                  <div class="text-lg">{analysisResult.value.explanation}</div>
                </div>
              </div>

              {/* Market Summary */}
              <div class="mb-8">
                <h3 class="text-xl font-semibold mb-4">Market Summary</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div class="bg-secondary p-4 rounded-lg">
                    <div class="text-sm text-neutral mb-1">Price Summary</div>
                    <div class="space-y-2">
                      <div>Last: ${analysisResult.value.analysis_details.market_summary.price_summary.last_price.toFixed(2)}</div>
                      <div>High 24h: ${analysisResult.value.analysis_details.market_summary.price_summary.price_high_24h.toFixed(2)}</div>
                      <div>Low 24h: ${analysisResult.value.analysis_details.market_summary.price_summary.price_low_24h.toFixed(2)}</div>
                    </div>
                  </div>
                  <div class="bg-secondary p-4 rounded-lg">
                    <div class="text-sm text-neutral mb-1">Volume Summary</div>
                    <div class="space-y-2">
                      <div>Avg 24h: ${analysisResult.value.analysis_details.market_summary.volume_summary.volume_average_24h.toFixed(2)}</div>
                      <div>High 24h: ${analysisResult.value.analysis_details.market_summary.volume_summary.volume_highest_24h.toFixed(2)}</div>
                    </div>
                  </div>
                  <div class="bg-secondary p-4 rounded-lg">
                    <div class="text-sm text-neutral mb-1">Market Sentiment</div>
                    <div class="space-y-2">
                      <div>Buy/Sell Ratio: {analysisResult.value.analysis_details.market_summary.market_sentiment.buy_sell_ratio?.toFixed(2) || 'N/A'}</div>
                      <div>Dominant Side: {analysisResult.value.analysis_details.market_summary.market_sentiment.dominant_side}</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Technical Indicators */}
              <div>
                <h3 class="text-xl font-semibold mb-4">Technical Indicators</h3>
                <div class="bg-secondary p-6 rounded-lg">
                  <div class="text-sm text-neutral mb-2 flex items-center justify-between">
                    <span>Full Technical Indicators Data</span>
                    <button 
                      class="p-1 rounded hover:bg-gray-700 transition-colors text-gray-300 hover:text-white flex items-center space-x-1"
                      onClick={() => {
                        if (analysisResult.value && analysisResult.value.analysis_details && analysisResult.value.analysis_details.technical_indicators) {
                          const btn = document.activeElement as HTMLElement;
                          const spanElement = btn.querySelector('span');
                          const originalText = spanElement ? spanElement.textContent || 'Copy' : 'Copy';
                          
                          if (spanElement) {
                            spanElement.textContent = 'Copying...';
                          }
                          
                          navigator.clipboard.writeText(JSON.stringify(analysisResult.value.analysis_details.technical_indicators, null, 2))
                            .then(() => {
                              if (spanElement) {
                                spanElement.textContent = 'Copied!';
                                setTimeout(() => {
                                  spanElement.textContent = originalText;
                                }, 2000);
                              }
                            })
                            .catch((error) => {
                              console.error('Failed to copy data: ', error);
                              if (spanElement) {
                                spanElement.textContent = 'Failed!';
                                setTimeout(() => {
                                  spanElement.textContent = originalText;
                                }, 2000);
                              }
                            });
                        }
                      }}
                      title="Copy to clipboard"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-12a2 2 0 00-2-2h-2M8 5a2 2 0 002-2h4a2 2 0 012 2M8 5a2 2 0 012-2h4a2 2 0 012 2" />
                      </svg>
                      <span class="text-xs">Copy</span>
                    </button>
                  </div>
                  <pre class="text-xs sm:text-sm overflow-auto max-h-[500px] bg-gray-800 text-gray-100 p-4 rounded whitespace-pre-wrap font-mono border border-gray-700 shadow-inner">
                    {"Available Indicator Keys: " + 
                      (analysisResult.value.analysis_details.technical_indicators ? 
                        Object.keys(analysisResult.value.analysis_details.technical_indicators).join(", ") : 
                        "None")}
                    {"\n\n"}
                    {JSON.stringify(analysisResult.value.analysis_details.technical_indicators, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}

          {showOrders.value && (
            <div class="mb-8">
              <h2 class="text-xl font-semibold text-primary mb-4">Recent Orders</h2>
              <OrderList orders={orders.value} />
            </div>
          )}

          {portfolioData.value && (
            <div class="portfolio-section bg-white rounded-lg shadow-lg p-6 mb-8">
              <h2 class="text-xl font-semibold text-primary mb-4">Your Portfolio</h2>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                {portfolioData.value.map((item: any) => (
                  <div class="bg-secondary p-4 rounded-lg">
                    <div class="text-sm text-neutral mb-1">{item.currency}</div>
                    <div class="text-lg font-semibold">${item.value_usd.toFixed(2)}</div>
                    <div class="text-sm text-neutral">Balance: {item.balance}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 