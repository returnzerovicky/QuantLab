/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo } from "react";
import {
  TrendingUp,
  TrendingDown,
  Folder,
  FolderOpen,
  FileCode,
  Terminal,
  Settings,
  Play,
  Download,
  Copy,
  Check,
  Briefcase,
  Cpu,
  BookOpen,
  ArrowRight,
  Activity,
  Percent,
  DollarSign,
  Layers,
  ChevronRight,
  Info,
  Clock,
  ExternalLink,
  Code
} from "lucide-react";
import JSZip from "jszip";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar
} from "recharts";
import { QUANTLAB_FILES } from "./quantlabFiles";

// Types for Simulated Price Data & Backtester
interface PriceRow {
  date: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  // Technical overlays
  smaFast?: number;
  smaSlow?: number;
  emaFast?: number;
  emaSlow?: number;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
  macdHist?: number;
}

interface BacktestResult {
  equityCurve: Array<{
    date: string;
    strategy: number;
    benchmark: number;
    drawdown: number;
    returns: number;
    price: number;
  }>;
  stats: {
    totalReturn: number;
    benchmarkReturn: number;
    cagr: number;
    volatility: number;
    sharpeRatio: number;
    sortinoRatio: number;
    maxDrawdown: number;
    tradesCount: number;
    winRate: number;
    commissionsPaid: number;
    finalCapital: number;
  };
  tradeLogs: Array<{
    date: string;
    type: "BUY" | "SELL" | "EXIT";
    price: number;
    equity: number;
    pnl?: number;
  }>;
}

// -------------------------------------------------------------
// SEED DATA GENERATOR (1 Year Daily Rows per Ticker)
// -------------------------------------------------------------
function generateHistoricalData(ticker: string, seed: number = 42): PriceRow[] {
  const dates: string[] = [];
  const start = new Date(2025, 0, 1); // 1-year historical dataset for 2025
  for (let i = 0; i < 260; i++) {
    // 260 trading days
    const d = new Date(start.getTime());
    d.setDate(start.getDate() + Math.floor(i * 1.4)); // Space out trading days
    if (d.getDay() !== 0 && d.getDay() !== 6) {
      dates.push(d.toISOString().split("T")[0]);
    } else {
      d.setDate(d.getDate() + 2);
      dates.push(d.toISOString().split("T")[0]);
    }
  }

  // Ticker specific behaviors
  let price = 150.0;
  let drift = 0.0005; // standard daily upward bias
  let vol = 0.015; // standard daily standard deviation
  let baseVolFactor = 1.0;

  switch (ticker) {
    case "AAPL":
      price = 180.0;
      drift = 0.0006;
      vol = 0.012;
      break;
    case "MSFT":
      price = 390.0;
      drift = 0.0008;
      vol = 0.011;
      break;
    case "TSLA":
      price = 220.0;
      drift = 0.0004;
      vol = 0.025;
      baseVolFactor = 1.5;
      break;
    case "BTC-USD":
      price = 45000.0;
      drift = 0.0015;
      vol = 0.038;
      baseVolFactor = 2.0;
      break;
    case "SPY":
      price = 480.0;
      drift = 0.0005;
      vol = 0.008;
      break;
  }

  // Pseudo-random generator (sine waves and noise)
  const rows: PriceRow[] = [];
  let currentPrice = price;

  for (let i = 0; i < dates.length; i++) {
    const cycle1 = Math.sin(i * 0.05) * 5.0 * baseVolFactor;
    const cycle2 = Math.cos(i * 0.15) * 2.0 * baseVolFactor;
    const randomNoise = (Math.sin(i * seed) * 0.5 + Math.cos(i * 17) * 0.5) * currentPrice * vol;
    
    // Geometric Brownian Motion step
    const change = currentPrice * drift + randomNoise + (cycle1 + cycle2) * 0.1;
    currentPrice = Math.max(1.0, currentPrice + change);

    const open = currentPrice - (Math.sin(i * 2) * 0.2) * (currentPrice * vol);
    const high = Math.max(open, currentPrice) + Math.abs(Math.cos(i * 13) * 0.5) * (currentPrice * vol);
    const low = Math.min(open, currentPrice) - Math.abs(Math.sin(i * 9) * 0.5) * (currentPrice * vol);
    const volume = Math.floor(1000000 + Math.abs(Math.sin(i) * 9000000) * baseVolFactor);

    rows.push({
      date: dates[i],
      close: parseFloat(currentPrice.toFixed(2)),
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      volume: volume
    });
  }

  return rows;
}

// -------------------------------------------------------------
// TECHNICAL INDICATORS ENGINE (JS PORT)
// -------------------------------------------------------------
function calculateIndicators(data: PriceRow[], params: {
  smaFast: number;
  smaSlow: number;
  emaFast: number;
  emaSlow: number;
  rsiPeriod: number;
  macdFast: number;
  macdSlow: number;
  macdSignal: number;
}): PriceRow[] {
  const result = data.map(item => ({ ...item }));
  const n = result.length;

  // 1. Simple Moving Averages
  for (let i = 0; i < n; i++) {
    if (i >= params.smaFast - 1) {
      let sum = 0;
      for (let k = 0; k < params.smaFast; k++) sum += result[i - k].close;
      result[i].smaFast = parseFloat((sum / params.smaFast).toFixed(2));
    }
    if (i >= params.smaSlow - 1) {
      let sum = 0;
      for (let k = 0; k < params.smaSlow; k++) sum += result[i - k].close;
      result[i].smaSlow = parseFloat((sum / params.smaSlow).toFixed(2));
    }
  }

  // 2. Exponential Moving Averages
  const calcEMA = (period: number, key: "emaFast" | "emaSlow") => {
    const k = 2 / (period + 1);
    let ema = result[0].close;
    result[0][key] = ema;
    for (let i = 1; i < n; i++) {
      ema = result[i].close * k + ema * (1 - k);
      result[i][key] = parseFloat(ema.toFixed(2));
    }
  };
  calcEMA(params.emaFast, "emaFast");
  calcEMA(params.emaSlow, "emaSlow");

  // 3. MACD
  const fastEMAKey = "emaFast_macd";
  const slowEMAKey = "emaSlow_macd";
  const kFast = 2 / (params.macdFast + 1);
  const kSlow = 2 / (params.macdSlow + 1);
  const kSignal = 2 / (params.macdSignal + 1);

  let emaFast = result[0].close;
  let emaSlow = result[0].close;
  for (let i = 0; i < n; i++) {
    emaFast = result[i].close * kFast + emaFast * (1 - kFast);
    emaSlow = result[i].close * kSlow + emaSlow * (1 - kSlow);
    result[i].macd = parseFloat((emaFast - emaSlow).toFixed(4));
  }

  let signal = result[0].macd || 0;
  for (let i = 0; i < n; i++) {
    const macdVal = result[i].macd || 0;
    signal = macdVal * kSignal + signal * (1 - kSignal);
    result[i].macdSignal = parseFloat(signal.toFixed(4));
    result[i].macdHist = parseFloat((macdVal - signal).toFixed(4));
  }

  // 4. RSI (Wilder's smoothed)
  let avgGain = 0;
  let avgLoss = 0;

  // First RSI value calculations
  for (let i = 1; i <= params.rsiPeriod; i++) {
    const change = result[i].close - result[i - 1].close;
    if (change > 0) avgGain += change;
    else avgLoss += Math.abs(change);
  }
  avgGain /= params.rsiPeriod;
  avgLoss /= params.rsiPeriod;

  if (params.rsiPeriod < n) {
    result[params.rsiPeriod].rsi = avgLoss === 0 ? 100 : parseFloat((100 - 100 / (1 + avgGain / avgLoss)).toFixed(2));
  }

  for (let i = params.rsiPeriod + 1; i < n; i++) {
    const change = result[i].close - result[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (params.rsiPeriod - 1) + gain) / params.rsiPeriod;
    avgLoss = (avgLoss * (params.rsiPeriod - 1) + loss) / params.rsiPeriod;

    result[i].rsi = avgLoss === 0 ? 100 : parseFloat((100 - 100 / (1 + avgGain / avgLoss)).toFixed(2));
  }

  return result;
}

// -------------------------------------------------------------
// STRATEGY SIMULATOR & BACKTEST ENGINE (JS PORT)
// -------------------------------------------------------------
function runQuantitativeBacktest(
  data: PriceRow[],
  strategy: "SMA" | "EMA" | "RSI" | "MACD",
  initialCapital: number = 10000.0,
  commissionPct: number = 0.001,
  strategyParams: {
    rsiOversold: number;
    rsiOverbought: number;
  }
): BacktestResult {
  const n = data.length;
  const equityCurve: BacktestResult["equityCurve"] = [];
  const tradeLogs: BacktestResult["tradeLogs"] = [];

  let cash = initialCapital;
  let position = 0; // 0 = Cash, 1 = Long Asset
  let currentEquity = initialCapital;
  let lastPosition = 0;

  const returnsArr: number[] = [];
  let peakEquity = initialCapital;
  let maxDD = 0;

  let tradeEntryPrice = 0;
  let tradeCount = 0;
  let winsCount = 0;
  let roundTrips = 0;
  let totalCommissions = 0;

  // Initialize first row
  equityCurve.push({
    date: data[0].date,
    strategy: initialCapital,
    benchmark: initialCapital,
    drawdown: 0,
    returns: 0,
    price: data[0].close
  });

  const benchmarkInitial = data[0].close;

  for (let i = 1; i < n; i++) {
    const prevRow = data[i - 1];
    const currentRow = data[i];

    // Determine signals (calculated at close of previous day i-1)
    let signal = 0; // 1 = Buy/Long, 0 = Sell/Cash

    if (strategy === "SMA") {
      const fast = prevRow.smaFast;
      const slow = prevRow.smaSlow;
      if (fast && slow) {
        signal = fast > slow ? 1 : 0;
      }
    } else if (strategy === "EMA") {
      const fast = prevRow.emaFast;
      const slow = prevRow.emaSlow;
      if (fast && slow) {
        signal = fast > slow ? 1 : 0;
      }
    } else if (strategy === "RSI") {
      const rsi = prevRow.rsi;
      if (rsi) {
        if (rsi < strategyParams.rsiOversold) {
          signal = 1; // Oversold rebound buy
        } else if (rsi > strategyParams.rsiOverbought) {
          signal = 0; // Overbought exit
        } else {
          // Keep current position
          signal = lastPosition;
        }
      }
    } else if (strategy === "MACD") {
      const macd = prevRow.macd;
      const signalLine = prevRow.macdSignal;
      if (macd !== undefined && signalLine !== undefined) {
        signal = macd > signalLine ? 1 : 0;
      }
    }

    // Execution Logic (Lag of 1-day applied to prevent foresight bias)
    // Position change occurs at the Close of today based on yesterday's signal
    const isTrade = signal !== position;
    let transactionFee = 0;

    if (isTrade) {
      tradeCount++;
      const tradeSize = currentEquity;
      transactionFee = tradeSize * commissionPct;
      totalCommissions += transactionFee;
      cash -= transactionFee;

      if (signal === 1) {
        // Enter Long
        position = 1;
        tradeEntryPrice = currentRow.close;
        tradeLogs.push({
          date: currentRow.date,
          type: "BUY",
          price: currentRow.close,
          equity: cash
        });
      } else {
        // Exit to Cash
        position = 0;
        roundTrips++;
        const pnlPct = (currentRow.close - tradeEntryPrice) / tradeEntryPrice;
        if (pnlPct > 0) winsCount++;
        
        tradeLogs.push({
          date: currentRow.date,
          type: "SELL",
          price: currentRow.close,
          equity: cash + currentEquity * pnlPct,
          pnl: pnlPct
        });
      }
    }

    // Daily equity valuation update
    const assetReturn = (currentRow.close - prevRow.close) / prevRow.close;
    const strategyReturn = position * assetReturn - (transactionFee / currentEquity);
    returnsArr.push(strategyReturn);

    currentEquity = currentEquity * (1 + strategyReturn);
    lastPosition = position;

    // Drawdown Calculation
    if (currentEquity > peakEquity) peakEquity = currentEquity;
    const dd = (currentEquity - peakEquity) / peakEquity;
    if (dd < maxDD) maxDD = dd;

    const benchmarkGrowth = initialCapital * (currentRow.close / benchmarkInitial);

    equityCurve.push({
      date: currentRow.date,
      strategy: parseFloat(currentEquity.toFixed(2)),
      benchmark: parseFloat(benchmarkGrowth.toFixed(2)),
      drawdown: parseFloat((dd * 100).toFixed(2)),
      returns: strategyReturn,
      price: currentRow.close
    });
  }

  // Calculate high-fidelity annualized statistics
  const totalReturn = (currentEquity / initialCapital) - 1;
  const benchmarkReturn = (data[n - 1].close / data[0].close) - 1;
  const years = n / 252;
  const cagr = Math.pow(currentEquity / initialCapital, 1 / years) - 1;

  // Volatility
  const avgReturn = returnsArr.reduce((s, x) => s + x, 0) / returnsArr.length;
  const variance = returnsArr.reduce((s, x) => s + Math.pow(x - avgReturn, 2), 0) / (returnsArr.length - 1);
  const dailyVol = Math.sqrt(variance);
  const annVol = dailyVol * Math.sqrt(252);

  // Sharpe (Risk free rate assumed at 2.0%)
  const rf = 0.02;
  const sharpeRatio = annVol > 0 ? (cagr - rf) / annVol : 0;

  // Downside Volatility for Sortino
  const downsideReturns = returnsArr.filter(x => x < 0);
  const downsideVariance = downsideReturns.reduce((s, x) => s + Math.pow(x - 0, 2), 0) / (returnsArr.length - 1);
  const downsideVol = Math.sqrt(downsideVariance) * Math.sqrt(252);
  const sortinoRatio = downsideVol > 0 ? (cagr - rf) / downsideVol : 0;

  const winRate = roundTrips > 0 ? winsCount / roundTrips : 0;

  return {
    equityCurve,
    stats: {
      totalReturn,
      benchmarkReturn,
      cagr,
      volatility: annVol,
      sharpeRatio,
      sortinoRatio,
      maxDrawdown: Math.abs(maxDD),
      tradesCount: tradeCount,
      winRate,
      commissionsPaid: totalCommissions,
      finalCapital: currentEquity
    },
    tradeLogs
  };
}

// -------------------------------------------------------------
// CORE REACT APPLICATION COMPONENT
// -------------------------------------------------------------
export default function App() {
  // Menu / Tab navigation
  const [activeTab, setActiveTab] = useState<"sandbox" | "codebase" | "guide">("sandbox");
  
  // Ticker choices
  const [selectedTicker, setSelectedTicker] = useState<"AAPL" | "MSFT" | "TSLA" | "BTC-USD" | "SPY">("AAPL");
  
  // Backtest strategy parameters
  const [selectedStrategy, setSelectedStrategy] = useState<"SMA" | "EMA" | "RSI" | "MACD">("SMA");
  const [startingCapital, setStartingCapital] = useState<number>(10000);
  const [commissionPercent, setCommissionPercent] = useState<number>(0.1); // in percent, e.g. 0.1% = 0.001
  
  // Advanced parameters sliders
  const [smaFast, setSmaFast] = useState<number>(20);
  const [smaSlow, setSmaSlow] = useState<number>(50);
  const [rsiOversold, setRsiOversold] = useState<number>(30);
  const [rsiOverbought, setRsiOverbought] = useState<number>(70);
  
  // Code Explorer States
  const [selectedCodeFile, setSelectedCodeFile] = useState<string>("app.py");
  const [copiedFile, setCopiedFile] = useState<boolean>(false);
  const [zippingState, setZippingState] = useState<"idle" | "zipping" | "success">("idle");

  // 1. Generate core stock data & calculate indicators
  const baseData = useMemo(() => {
    return generateHistoricalData(selectedTicker);
  }, [selectedTicker]);

  const dataWithIndicators = useMemo(() => {
    return calculateIndicators(baseData, {
      smaFast,
      smaSlow,
      emaFast: 9,
      emaSlow: 21,
      rsiPeriod: 14,
      macdFast: 12,
      macdSlow: 26,
      macdSignal: 9
    });
  }, [baseData, smaFast, smaSlow]);

  // 2. Run active strategy backtest
  const backtestResult = useMemo(() => {
    return runQuantitativeBacktest(
      dataWithIndicators,
      selectedStrategy,
      startingCapital,
      commissionPercent / 100,
      { rsiOversold, rsiOverbought }
    );
  }, [dataWithIndicators, selectedStrategy, startingCapital, commissionPercent, rsiOversold, rsiOverbought]);

  // 3. Trigger standard zip downloader
  const handleDownloadZip = async () => {
    setZippingState("zipping");
    const zip = new JSZip();

    // Create primary folders
    zip.file("README.md", QUANTLAB_FILES["README.md"].content);
    zip.file("requirements.txt", QUANTLAB_FILES["requirements.txt"].content);
    zip.file(".gitignore", QUANTLAB_FILES[".gitignore"].content);
    zip.file("app.py", QUANTLAB_FILES["app.py"].content);

    const srcFolder = zip.folder("src");
    if (srcFolder) {
      srcFolder.file("__init__.py", QUANTLAB_FILES["src/__init__.py"].content);
      srcFolder.file("data_loader.py", QUANTLAB_FILES["src/data_loader.py"].content);
      srcFolder.file("indicators.py", QUANTLAB_FILES["src/indicators.py"].content);
      srcFolder.file("metrics.py", QUANTLAB_FILES["src/metrics.py"].content);
      srcFolder.file("strategies.py", QUANTLAB_FILES["src/strategies.py"].content);
      srcFolder.file("portfolio.py", QUANTLAB_FILES["src/portfolio.py"].content);
      srcFolder.file("backtester.py", QUANTLAB_FILES["src/backtester.py"].content);
      srcFolder.file("visualizer.py", QUANTLAB_FILES["src/visualizer.py"].content);
      srcFolder.file("utils.py", QUANTLAB_FILES["src/utils.py"].content);
    }

    // Add empty folders required by architecture
    zip.folder("data");
    zip.folder("assets");
    zip.folder("notebooks");

    try {
      const content = await zip.generateAsync({ type: "blob" });
      const url = window.URL.createObjectURL(content);
      const link = document.createElement("a");
      link.href = url;
      link.download = "QuantLab-main.zip";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setZippingState("success");
      setTimeout(() => setZippingState("idle"), 3000);
    } catch (err) {
      setZippingState("idle");
    }
  };

  // Copy code utility
  const handleCopyCode = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  // Quick statistics
  const formatPct = (val: number) => {
    return (val * 100).toFixed(2) + "%";
  };

  const formatCurrency = (val: number) => {
    return "$" + val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  return (
    <div className="min-h-screen bg-[#080d1a] text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-black">
      {/* -------------------------------------------------------------
          TOP ACTION BAR & GLASS HEADER
         ------------------------------------------------------------- */}
      <header className="border-b border-slate-800 bg-[#0c1325]/90 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-emerald-500/10 border border-emerald-500/20 p-2 rounded-lg text-emerald-400">
              <Terminal size={22} className="animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">QuantLab</h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  ACTIVE PREVIEW
                </span>
              </div>
              <p className="text-xs text-slate-400">Production-Ready Python Quantitative Backtesting & Streamlit Platform</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleDownloadZip}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-xs transition duration-200 shadow-md ${
                zippingState === "success"
                  ? "bg-emerald-600 text-white"
                  : "bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-black font-semibold hover:shadow-emerald-500/10"
              }`}
            >
              {zippingState === "zipping" ? (
                <>
                  <Clock className="animate-spin" size={14} />
                  Bundling Repo...
                </>
              ) : zippingState === "success" ? (
                <>
                  <Check size={14} />
                  Downloaded Repo!
                </>
              ) : (
                <>
                  <Download size={14} />
                  Download Python Repo (.ZIP)
                </>
              )}
            </button>
            <a
              href={QUANTLAB_FILES["README.md"] ? "#" : undefined}
              onClick={(e) => {
                e.preventDefault();
                setActiveTab("codebase");
              }}
              className="px-3 py-2 border border-slate-700 hover:border-slate-600 hover:bg-slate-800/40 rounded-lg text-xs font-medium text-slate-300 transition"
            >
              Browse Code
            </a>
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------------
          TAB CONTROLS & SUB-BAR
         ------------------------------------------------------------- */}
      <div className="bg-[#0b1020] border-b border-slate-800 py-1.5 px-4 sticky top-[69px] z-40">
        <div className="max-w-7xl mx-auto flex gap-1">
          <button
            onClick={() => setActiveTab("sandbox")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition duration-150 ${
              activeTab === "sandbox"
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Layers size={14} />
            Quant Interactive Station
          </button>
          <button
            onClick={() => setActiveTab("codebase")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition duration-150 ${
              activeTab === "codebase"
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Folder size={14} />
            Python File Explorer
          </button>
          <button
            onClick={() => setActiveTab("guide")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition duration-150 ${
              activeTab === "guide"
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BookOpen size={14} />
            Deployment & Setup
          </button>
        </div>
      </div>

      {/* -------------------------------------------------------------
          MAIN TAB CONTENT AREA
         ------------------------------------------------------------- */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">
        {/* TAB 1: INTERACTIVE QUANTLAB STATION */}
        {activeTab === "sandbox" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* LEFT SIDEBAR CONTROLS */}
            <div className="lg:col-span-3 flex flex-col gap-5">
              <div className="bg-[#0c1325] border border-slate-800 rounded-xl p-5 shadow-xl">
                <div className="flex items-center gap-2 text-emerald-400 font-semibold text-xs uppercase tracking-wider mb-4">
                  <Settings size={14} />
                  Simulation Inputs
                </div>

                <div className="flex flex-col gap-4">
                  {/* Ticker Selector */}
                  <div>
                    <label className="block text-slate-400 text-[10px] uppercase font-semibold tracking-wider mb-1.5">
                      Target Asset
                    </label>
                    <select
                      value={selectedTicker}
                      onChange={(e) => setSelectedTicker(e.target.value as any)}
                      className="w-full text-xs bg-[#070b16] border border-slate-700 rounded-lg py-2 px-2.5 text-white focus:outline-none focus:border-emerald-500 transition"
                    >
                      <option value="AAPL">AAPL (Apple Blue-Chip)</option>
                      <option value="MSFT">MSFT (Microsoft Momentum)</option>
                      <option value="TSLA">TSLA (Tesla High Beta)</option>
                      <option value="BTC-USD">BTC-USD (Crypto Volatile)</option>
                      <option value="SPY">SPY (S&P 500 Benchmark)</option>
                    </select>
                  </div>

                  {/* Strategy Choice */}
                  <div>
                    <label className="block text-slate-400 text-[10px] uppercase font-semibold tracking-wider mb-1.5">
                      Backtest Strategy
                    </label>
                    <select
                      value={selectedStrategy}
                      onChange={(e) => setSelectedStrategy(e.target.value as any)}
                      className="w-full text-xs bg-[#070b16] border border-slate-700 rounded-lg py-2 px-2.5 text-white focus:outline-none focus:border-emerald-500 transition"
                    >
                      <option value="SMA">SMA Crossover (Trend)</option>
                      <option value="EMA">EMA Crossover (Speed)</option>
                      <option value="RSI">RSI Overbought/Oversold</option>
                      <option value="MACD">MACD Signal Crossover</option>
                    </select>
                  </div>

                  {/* Financial variables */}
                  <div>
                    <label className="block text-slate-400 text-[10px] uppercase font-semibold tracking-wider mb-1.5">
                      Starting capital
                    </label>
                    <div className="relative">
                      <span className="absolute left-2.5 top-2 text-slate-500 text-xs">$</span>
                      <input
                        type="number"
                        value={startingCapital}
                        onChange={(e) => setStartingCapital(Math.max(100, Number(e.target.value)))}
                        className="w-full text-xs bg-[#070b16] border border-slate-700 rounded-lg py-2 pl-6 pr-2.5 text-white focus:outline-none focus:border-emerald-500 transition"
                      />
                    </div>
                  </div>

                  {/* Slippage & Friction */}
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="text-slate-400 text-[10px] uppercase font-semibold tracking-wider">
                        Broker Fee & Slippage
                      </label>
                      <span className="text-[10px] font-mono text-emerald-400 font-semibold">{commissionPercent}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.05"
                      value={commissionPercent}
                      onChange={(e) => setCommissionPercent(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500 cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
                    />
                  </div>

                  <hr className="border-slate-800" />

                  {/* Strategy Specific Hyperparameters */}
                  <div className="bg-[#070b16]/60 p-3 rounded-lg border border-slate-800/80 flex flex-col gap-3">
                    <span className="text-[10px] text-emerald-400 font-semibold uppercase tracking-wider">
                      Strategy Hyperparameters
                    </span>

                    {(selectedStrategy === "SMA" || selectedStrategy === "EMA") && (
                      <>
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span>Fast MA Period</span>
                            <span className="font-mono text-white">{smaFast}d</span>
                          </div>
                          <input
                            type="range"
                            min="5"
                            max="50"
                            value={smaFast}
                            onChange={(e) => setSmaFast(parseInt(e.target.value))}
                            className="w-full accent-emerald-500 h-1 bg-slate-800 rounded-lg"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span>Slow MA Period</span>
                            <span className="font-mono text-white">{smaSlow}d</span>
                          </div>
                          <input
                            type="range"
                            min="30"
                            max="150"
                            value={smaSlow}
                            onChange={(e) => setSmaSlow(parseInt(e.target.value))}
                            className="w-full accent-emerald-500 h-1 bg-slate-800 rounded-lg"
                          />
                        </div>
                      </>
                    )}

                    {selectedStrategy === "RSI" && (
                      <>
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span>Oversold Buy Threshold</span>
                            <span className="font-mono text-white">{rsiOversold}</span>
                          </div>
                          <input
                            type="range"
                            min="15"
                            max="45"
                            value={rsiOversold}
                            onChange={(e) => setRsiOversold(parseInt(e.target.value))}
                            className="w-full accent-emerald-500 h-1 bg-slate-800 rounded-lg"
                          />
                        </div>
                        <div>
                          <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                            <span>Overbought Sell Threshold</span>
                            <span className="font-mono text-white">{rsiOverbought}</span>
                          </div>
                          <input
                            type="range"
                            min="55"
                            max="85"
                            value={rsiOverbought}
                            onChange={(e) => setRsiOverbought(parseInt(e.target.value))}
                            className="w-full accent-emerald-500 h-1 bg-slate-800 rounded-lg"
                          />
                        </div>
                      </>
                    )}

                    {selectedStrategy === "MACD" && (
                      <p className="text-[10px] text-slate-400 leading-relaxed">
                        Industry Standard settings applied: Fast EMA (12), Slow EMA (26), Signal Line SMA (9).
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* Informative Help Card */}
              <div className="bg-slate-900/40 border border-slate-800/80 p-4 rounded-xl flex gap-3 text-slate-300">
                <Info size={28} className="text-emerald-400 shrink-0" />
                <div className="text-[11px] leading-relaxed">
                  <strong className="text-white block mb-0.5">Execution Realism Enabled</strong>
                  Signal calculates on Close(t) and triggers at Close(t+1) (1-day execution lag). This mathematically ensures zero lookback / lookahead bias, reflecting production-grade algorithmic standards.
                </div>
              </div>
            </div>

            {/* RIGHT WORKSPACE PANELS */}
            <div className="lg:col-span-9 flex flex-col gap-6">
              
              {/* BACKTEST METRICS STATS SUMMARY */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-[#0c1325] border border-slate-800/80 p-4 rounded-xl shadow-md">
                  <div className="flex items-center gap-1.5 text-slate-400 text-[10px] font-semibold uppercase tracking-wider mb-1">
                    <Percent size={12} className="text-emerald-400" />
                    Strategy Return
                  </div>
                  <div className="text-xl font-bold font-mono text-white flex items-center gap-1.5">
                    {formatPct(backtestResult.stats.totalReturn)}
                    <span className="text-[10px] font-normal text-slate-400">
                      (vs {formatPct(backtestResult.stats.benchmarkReturn)} B&H)
                    </span>
                  </div>
                </div>

                <div className="bg-[#0c1325] border border-slate-800/80 p-4 rounded-xl shadow-md">
                  <div className="flex items-center gap-1.5 text-slate-400 text-[10px] font-semibold uppercase tracking-wider mb-1">
                    <TrendingUp size={12} className="text-blue-400" />
                    Annualized CAGR
                  </div>
                  <div className="text-xl font-bold font-mono text-white">
                    {formatPct(backtestResult.stats.cagr)}
                  </div>
                </div>

                <div className="bg-[#0c1325] border border-slate-800/80 p-4 rounded-xl shadow-md">
                  <div className="flex items-center gap-1.5 text-slate-400 text-[10px] font-semibold uppercase tracking-wider mb-1">
                    <Briefcase size={12} className="text-purple-400" />
                    Sharpe & Sortino
                  </div>
                  <div className="text-xl font-bold font-mono text-white">
                    {backtestResult.stats.sharpeRatio.toFixed(2)}
                    <span className="text-xs text-slate-500 ml-1">/ {backtestResult.stats.sortinoRatio.toFixed(2)}</span>
                  </div>
                </div>

                <div className="bg-[#0c1325] border border-slate-800/80 p-4 rounded-xl shadow-md">
                  <div className="flex items-center gap-1.5 text-slate-400 text-[10px] font-semibold uppercase tracking-wider mb-1">
                    <TrendingDown size={12} className="text-rose-400" />
                    Max Drawdown
                  </div>
                  <div className="text-xl font-bold font-mono text-rose-400">
                    -{formatPct(backtestResult.stats.maxDrawdown)}
                  </div>
                </div>
              </div>

              {/* ACTIVE WORKSPACE CHARTS */}
              <div className="bg-[#0c1325] border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-6">
                <div>
                  <h3 className="text-sm font-semibold text-white mb-1">Portfolio Equity Growth ($10,000 Starting Base)</h3>
                  <p className="text-[11px] text-slate-400">Comparing active algorithmic strategy results (green) against Buy & Hold benchmark indices (gray).</p>
                </div>

                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={backtestResult.equityCurve} margin={{ top: 5, right: 5, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" stroke="#64748b" fontSize={10} minTickGap={40} />
                      <YAxis stroke="#64748b" fontSize={10} tickFormatter={(val) => `$${val.toLocaleString()}`} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "8px" }}
                        labelClassName="text-[10px] font-mono text-slate-400"
                        formatter={(value: any) => [`$${parseFloat(value).toLocaleString()}`, "Equity"]}
                      />
                      <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: "11px" }} />
                      <Line
                        type="monotone"
                        dataKey="strategy"
                        name={`${selectedStrategy} Strategy (Net of Friction)`}
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="benchmark"
                        name="Buy & Hold Benchmark"
                        stroke="#64748b"
                        strokeWidth={1.5}
                        strokeDasharray="4 4"
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* ASSET PRICE ACTION & INDICATOR OVERLAY */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Price Action Plot */}
                <div className="bg-[#0c1325] border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-xs font-semibold text-white mb-1.5 uppercase tracking-wide">
                    Asset Price action & Technical Overlays
                  </h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={dataWithIndicators} margin={{ top: 5, right: 5, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" stroke="#64748b" fontSize={9} minTickGap={40} />
                        <YAxis stroke="#64748b" fontSize={9} domain={["auto", "auto"]} />
                        <Tooltip
                          contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "8px" }}
                          formatter={(val: any) => [`$${parseFloat(val).toFixed(2)}`, "Price"]}
                        />
                        <Line type="monotone" dataKey="close" name="Close" stroke="#f1f5f9" strokeWidth={1.5} dot={false} />
                        {(selectedStrategy === "SMA" || selectedStrategy === "EMA") && (
                          <>
                            <Line
                              type="monotone"
                              dataKey={selectedStrategy === "SMA" ? "smaFast" : "emaFast"}
                              name="Fast MA"
                              stroke="#3b82f6"
                              strokeWidth={1.2}
                              dot={false}
                            />
                            <Line
                              type="monotone"
                              dataKey={selectedStrategy === "SMA" ? "smaSlow" : "emaSlow"}
                              name="Slow MA"
                              stroke="#f59e0b"
                              strokeWidth={1.2}
                              dot={false}
                            />
                          </>
                        )}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Drawdowns Plot */}
                <div className="bg-[#0c1325] border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-xs font-semibold text-white mb-1.5 uppercase tracking-wide">
                    Strategy Drawdown Profile (%)
                  </h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={backtestResult.equityCurve} margin={{ top: 5, right: 5, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" stroke="#64748b" fontSize={9} minTickGap={40} />
                        <YAxis stroke="#64748b" fontSize={9} tickFormatter={(val) => `${val}%`} />
                        <Tooltip
                          contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "8px" }}
                          formatter={(val: any) => [`${val}%`, "Drawdown"]}
                        />
                        <Area
                          type="monotone"
                          dataKey="drawdown"
                          name="Drawdown"
                          stroke="#ef4444"
                          fill="rgba(239, 68, 68, 0.15)"
                          strokeWidth={1.2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* SECONDARY STATS ROW & RECENT TRADE LOGS */}
              <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                
                {/* Backtester Trade Stats Card */}
                <div className="md:col-span-5 bg-[#0c1325] border border-slate-800 rounded-xl p-5 flex flex-col gap-4">
                  <span className="text-xs font-semibold text-white uppercase tracking-wider">
                    Execution Logs Analytics
                  </span>

                  <div className="flex flex-col gap-3 text-xs">
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-400">Positions Executed</span>
                      <span className="font-mono text-white font-semibold">{backtestResult.stats.tradesCount} trades</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-400">Strategy Win Rate</span>
                      <span className="font-mono text-emerald-400 font-semibold">{(backtestResult.stats.winRate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800 pb-2">
                      <span className="text-slate-400">Total Commissions Paid</span>
                      <span className="font-mono text-rose-400 font-semibold">{formatCurrency(backtestResult.stats.commissionsPaid)}</span>
                    </div>
                    <div className="flex justify-between pb-1">
                      <span className="text-slate-400">Starting Portfolio Capital</span>
                      <span className="font-mono text-slate-300">{formatCurrency(startingCapital)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-semibold text-white">Ending Capital Value</span>
                      <span className="font-mono text-emerald-400 font-bold">{formatCurrency(backtestResult.stats.finalCapital)}</span>
                    </div>
                  </div>

                  <div className="p-3 bg-emerald-500/5 rounded-lg border border-emerald-500/10 text-[11px] leading-relaxed text-emerald-300">
                    🔬 <strong>Statistical Output Verdict</strong>: This strategy demonstrates a Sharpe of{" "}
                    <strong>{backtestResult.stats.sharpeRatio.toFixed(2)}</strong> and completed{" "}
                    <strong>{backtestResult.stats.tradesCount}</strong> operations. Download the Python codebase to run high-density grid optimization across multiple assets locally.
                  </div>
                </div>

                {/* Trades Logs */}
                <div className="md:col-span-7 bg-[#0c1325] border border-slate-800 rounded-xl p-5 flex flex-col gap-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-semibold text-white uppercase tracking-wider">
                      Recent Signal Ledger Entries
                    </span>
                    <span className="text-[10px] text-slate-500">Showing last 5 operations</span>
                  </div>

                  <div className="flex-1 overflow-x-auto">
                    <table className="w-full text-left text-[11px]">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400">
                          <th className="py-2">Date</th>
                          <th className="py-2">Order Type</th>
                          <th className="py-2">Asset Price</th>
                          <th className="py-2">Cash Base</th>
                          <th className="py-2 text-right">Trade PnL</th>
                        </tr>
                      </thead>
                      <tbody>
                        {backtestResult.tradeLogs.length === 0 ? (
                          <tr>
                            <td colSpan={5} className="py-4 text-center text-slate-500 italic">
                              No trades executed by strategy parameters. Maintaining 100% Cash holding.
                            </td>
                          </tr>
                        ) : (
                          backtestResult.tradeLogs.slice(-5).reverse().map((log, idx) => (
                            <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/10">
                              <td className="py-2 font-mono text-slate-400">{log.date}</td>
                              <td className="py-2">
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${
                                    log.type === "BUY"
                                      ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                      : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                                  }`}
                                >
                                  {log.type}
                                </span>
                              </td>
                              <td className="py-2 font-mono text-slate-300">${log.price.toFixed(2)}</td>
                              <td className="py-2 font-mono text-slate-300">${log.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                              <td className={`py-2 text-right font-mono font-semibold ${
                                log.pnl !== undefined
                                  ? log.pnl > 0
                                    ? "text-emerald-400"
                                    : "text-rose-400"
                                  : "text-slate-400"
                              }`}>
                                {log.pnl !== undefined ? (log.pnl > 0 ? "+" : "") + (log.pnl * 100).toFixed(2) + "%" : "--"}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

              </div>

            </div>
          </div>
        )}

        {/* TAB 2: IDE-STYLE PYTHON CODEBASE EXPLORER */}
        {activeTab === "codebase" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 border border-slate-800 rounded-xl overflow-hidden shadow-2xl bg-[#0c1325]">
            
            {/* FILE NAVIGATION TREE */}
            <div className="lg:col-span-3 border-r border-slate-800 bg-[#090e1a] p-4 flex flex-col gap-4">
              <div>
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-2">
                  QuantLab Directory Tree
                </span>
                <div className="flex items-center gap-2 text-slate-300 font-semibold text-xs mb-1">
                  <FolderOpen size={16} className="text-emerald-400" />
                  <span>QuantLab/</span>
                </div>
              </div>

              {/* Core Root Files */}
              <div className="flex flex-col gap-1 pl-4 text-xs">
                {Object.keys(QUANTLAB_FILES).filter(k => !k.startsWith("src/")).map(fileName => (
                  <button
                    key={fileName}
                    onClick={() => setSelectedCodeFile(fileName)}
                    className={`flex items-center gap-2 py-1.5 px-2 rounded-md transition text-left ${
                      selectedCodeFile === fileName
                        ? "bg-slate-800/70 text-emerald-400 font-medium border-l-2 border-emerald-500"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    <FileCode size={13} className="text-emerald-500/70" />
                    <span className="truncate">{fileName}</span>
                  </button>
                ))}
              </div>

              {/* src/ folder sub-branch */}
              <div className="flex flex-col gap-1 pl-4">
                <div className="flex items-center gap-2 text-slate-400 font-medium text-xs mb-1">
                  <Folder size={14} className="text-blue-400" />
                  <span>src/</span>
                </div>
                <div className="flex flex-col gap-1 pl-4 text-xs">
                  {Object.keys(QUANTLAB_FILES).filter(k => k.startsWith("src/")).map(filePath => {
                    const cleanName = filePath.replace("src/", "");
                    return (
                      <button
                        key={filePath}
                        onClick={() => setSelectedCodeFile(filePath)}
                        className={`flex items-center gap-2 py-1.5 px-2 rounded-md transition text-left ${
                          selectedCodeFile === filePath
                            ? "bg-slate-800/70 text-emerald-400 font-medium border-l-2 border-emerald-500"
                            : "text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <FileCode size={13} className="text-blue-500/70" />
                        <span className="truncate">{cleanName}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="border-t border-slate-800/80 pt-4 mt-auto">
                <div className="p-3 bg-slate-900/60 rounded-lg text-[10px] text-slate-400 leading-relaxed border border-slate-800/60">
                  ⚡ <strong>Developer Tip</strong>: You can click the "Download ZIP" button in the header to export this exact, fully assembled repository structures locally instantly.
                </div>
              </div>
            </div>

            {/* HIGH-FIDELITY CODE EDITOR CONTAINER */}
            <div className="lg:col-span-9 bg-[#0c1325] flex flex-col h-[650px]">
              
              {/* CODE EDITOR HEADER BAR */}
              <div className="bg-[#090e1a] border-b border-slate-800 px-5 py-3 flex justify-between items-center">
                <div className="flex items-center gap-2.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  <span className="text-xs font-mono text-slate-400 ml-2">
                    QuantLab / {selectedCodeFile}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleCopyCode(QUANTLAB_FILES[selectedCodeFile].content)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition duration-150"
                  >
                    {copiedFile ? (
                      <>
                        <Check size={12} className="text-emerald-400" />
                        <span className="text-emerald-400">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy size={12} />
                        <span>Copy File</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* CODE DISPLAY AREA (SYNTAX EMULATOR WITH LINE NUMBERS) */}
              <div className="flex-1 overflow-y-auto p-5 font-mono text-[11px] leading-relaxed bg-[#070b16] text-slate-300 select-text">
                <pre className="flex gap-4">
                  {/* Fake Line Numbers for IDE Realism */}
                  <div className="text-slate-600 text-right select-none pr-3 border-r border-slate-800/80">
                    {QUANTLAB_FILES[selectedCodeFile].content.split("\n").map((_, i) => (
                      <div key={i}>{i + 1}</div>
                    ))}
                  </div>

                  {/* Highlight Emulator */}
                  <div className="text-slate-300 overflow-x-auto whitespace-pre">
                    {QUANTLAB_FILES[selectedCodeFile].content}
                  </div>
                </pre>
              </div>

            </div>

          </div>
        )}

        {/* TAB 3: DEPLOYMENT AND USAGE GUIDE */}
        {activeTab === "guide" && (
          <div className="flex flex-col gap-6 max-w-4xl mx-auto">
            <div className="bg-[#0c1325] border border-slate-800 p-6 rounded-xl shadow-md">
              <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Cpu size={20} className="text-emerald-400" />
                Setting Up & Running QuantLab Locally
              </h2>
              <p className="text-slate-400 text-xs mb-4 leading-relaxed">
                QuantLab is architected to run as a native Python Streamlit application with a complete backend technical calculator.
                Follow these simple steps to run the Streamlit dashboard on your computer.
              </p>

              <hr className="border-slate-800 mb-5" />

              <div className="flex flex-col gap-5 text-xs leading-relaxed">
                <div>
                  <strong className="text-emerald-400 block mb-1">1. Set Up Python Virtual Environment</strong>
                  <p className="text-slate-300 mb-2">Create a clean virtual environment to prevent package version conflicts on your local system.</p>
                  <pre className="bg-[#070b16] border border-slate-800 p-3 rounded-lg text-[11px] text-slate-300 font-mono">
                    python3 -m venv venv{"\n"}
                    source venv/bin/activate  # Windows users: venv\Scripts\activate
                  </pre>
                </div>

                <div>
                  <strong className="text-emerald-400 block mb-1">2. Install Package Dependencies</strong>
                  <p className="text-slate-300 mb-2">Install all optimized math, pandas, plotting and retrieval packages listed in requirements.txt.</p>
                  <pre className="bg-[#070b16] border border-slate-800 p-3 rounded-lg text-[11px] text-slate-300 font-mono">
                    pip install --upgrade pip{"\n"}
                    pip install -r requirements.txt
                  </pre>
                </div>

                <div>
                  <strong className="text-emerald-400 block mb-1">3. Start the Interactive Streamlit Dashboard</strong>
                  <p className="text-slate-300 mb-2">Boot the visual research laboratory dashboard server with a single terminal instruction.</p>
                  <pre className="bg-[#070b16] border border-slate-800 p-3 rounded-lg text-[11px] text-slate-300 font-mono">
                    streamlit run app.py
                  </pre>
                  <span className="text-[10px] text-slate-500 block mt-1">
                    The command will output: "Local URL: http://localhost:8501" and open the lab automatically.
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-[#0c1325] border border-slate-800 p-5 rounded-xl">
                <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-1.5">
                  <Activity size={16} className="text-blue-400" />
                  Extending with Custom Strategies
                </h3>
                <p className="text-slate-400 text-xs leading-relaxed mb-3">
                  You can easily write your own trading rules by appending formulas to <code>src/strategies.py</code>. Just return a Pandas Series containing <code>1.0</code> for long position triggers, <code>-1.0</code> for short, and <code>0.0</code> for exit.
                </p>
                <div className="text-[10px] text-emerald-400 font-mono bg-emerald-950/20 p-2.5 rounded border border-emerald-800/30">
                  def my_custom_strategy(df):{"\n"}
                  &nbsp;&nbsp;&nbsp;&nbsp;# Custom logic here{"\n"}
                  &nbsp;&nbsp;&nbsp;&nbsp;df["Signal"] = df["RSI_14"] &lt; 25{"\n"}
                  &nbsp;&nbsp;&nbsp;&nbsp;return df
                </div>
              </div>

              <div className="bg-[#0c1325] border border-slate-800 p-5 rounded-xl flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white mb-2 flex items-center gap-1.5">
                    <Layers size={16} className="text-purple-400" />
                    Machine Learning Predictive Power
                  </h3>
                  <p className="text-slate-400 text-xs leading-relaxed mb-4">
                    QuantLab has built-in Scikit-Learn utility helper functions. By generating lag targets inside <code>src/utils.py</code>, you can train Random Forest regression or SVM classifiers locally and use them to predict next-day pricing directions.
                  </p>
                </div>
                <div className="p-3 bg-slate-900/50 border border-slate-800 rounded-lg text-[10px] text-slate-300">
                  💡 <strong>Scikit-Learn splitters</strong> ensure walk-forward temporal cross-validation prevents historical training leakages.
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* -------------------------------------------------------------
          FOOTER CREDIT SYSTEM
         ------------------------------------------------------------- */}
      <footer className="border-t border-slate-900 bg-[#060a14] py-6 mt-12 text-slate-500 text-xs">
        <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-4">
          <div>
            <span>QuantLab Quantitative Station. Created in Cloud Container Workspace.</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>Python 3.11</span>
            <span>Streamlit Engine</span>
            <span>yfinance & Pandas</span>
            <span>Scikit-Learn Ready</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
