"""
Technical Analysis Engine
Contains functions for performing technical analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from app.technical_indicators.basic_indicators import (
    macd, bollinger_bands, relative_strength_index, average_true_range,
    pivot_points, elder_ray_index, stochastic_oscillator, on_balance_volume,
    keltner_channel, directional_movement_index, williams_percent_r, 
    volume_zone_oscillator, relative_volume, chandelier_exit, market_regime
)
from app.technical_indicators.advanced_indicators import (
    accumulation_distribution_line, chaikin_oscillator, aroon_indicator,
    chaikin_money_flow, parabolic_sar, money_flow_index, percentage_price_oscillator,
    donchian_channels, rate_of_change, commodity_channel_index, awesome_oscillator,
    vortex_indicator, true_strength_index, mass_index, hull_moving_average,
    coppock_curve, klinger_oscillator, ichimoku_cloud, supertrend, heikin_ashi,
    camarilla_pivot_points, woodie_pivot_points, demark_pivot_points, squeeze_momentum,
    ehlers_fisher_transform, chande_momentum_oscillator, elder_triple_screen,
    volume_profile, harmonic_patterns, divergence_scanner,
    stochastic_rsi, elliott_wave_tracker, mean_reversion_index,
    market_breadth_indicators, orderflow_analysis
)

# Configure logging
logger = logging.getLogger(__name__)

def calculate_technical_indicators(pair_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate various technical indicators based on the pair data
    """
    # Check if we have candle data
    if not pair_data or 'candles' not in pair_data or not pair_data['candles']:
        logger.warning("No candle data available for technical analysis")
        return {}
    
    # Convert candles to DataFrame for easier manipulation
    candles = pd.DataFrame(pair_data['candles'])
    
    # Check if candles DataFrame is empty or doesn't have required columns
    required_columns = ['start', 'open', 'high', 'low', 'close', 'volume']
    if candles.empty or not all(col in candles.columns for col in required_columns):
        logger.warning("Candle data is missing required columns")
        return {}
    
    # Log the data length to help with debugging
    data_length = len(candles)
    logger.info(f"Processing technical analysis with {data_length} candles")
    
    # Check if we have sufficient data for meaningful technical analysis
    if data_length < 5:
        logger.warning("Insufficient data for technical analysis: fewer than 5 candles available")
        return {"error": "Insufficient data for technical analysis: fewer than 5 candles available"}
    
    # Additional data quality check for problematic values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if candles[col].isnull().any():
            logger.warning(f"Data contains NULL values in {col} column, filling with forward fill method")
            candles[col] = candles[col].fillna(method='ffill')
        
        if (candles[col] <= 0).any() and col != 'volume':
            logger.warning(f"Data contains zero or negative values in {col} column, which may cause calculation errors")
    
    # Sort by timestamp
    candles = candles.sort_values('start')
    
    # Use parallel processing to calculate indicators
    return calculate_technical_indicators_parallel(candles)

def calculate_technical_indicators_parallel(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Use parallel processing to calculate all technical indicators
    """
    # Function to safely get value from series
    def safe_get_value(series, default=np.nan):
        if series is None or isinstance(series, (int, float)):
            return default
        if isinstance(series, pd.Series) and series.empty:
            return default
        if isinstance(series, list) and not series:
            return default
        if isinstance(series, np.ndarray) and len(series) == 0:
            return default
        try:
            if isinstance(series, pd.Series):
                return series.iloc[-1]
            elif isinstance(series, (list, np.ndarray)) and len(series) > 0:
                return series[-1]
            return default
        except Exception:
            return default
    
    # Create dictionaries to store all indicator results
    all_indicators = {}
    result = {}  # Initialize the result dictionary
    
    # Calculate Moving Averages
    def calculate_moving_averages():
        try:
            ma_periods = [5, 10, 20, 50, 100, 200]
            ema_periods = [5, 10, 20, 50, 100, 200]
            
            # Simple Moving Averages (SMA)
            sma_results = {}
            for period in ma_periods:
                try:
                    # Ensure we have enough data for the calculation
                    if len(df) < period:
                        sma_results[f"SMA_{period}"] = {
                            "value": np.nan,
                            "data": [],
                            "period": period
                        }
                        continue
                        
                    sma = df['close'].rolling(window=period).mean()
                    sma_results[f"SMA_{period}"] = {
                        "value": safe_get_value(sma),
                        "data": sma.values.tolist() if len(sma) > 0 else [],
                        "period": period
                    }
                except Exception as e:
                    logger.error(f"Error calculating SMA_{period}: {str(e)}")
                    sma_results[f"SMA_{period}"] = {
                        "value": np.nan,
                        "data": [],
                        "period": period,
                        "error": str(e)
                    }
            
            # Exponential Moving Averages (EMA)
            ema_results = {}
            for period in ema_periods:
                try:
                    # Ensure we have enough data for the calculation
                    if len(df) < period:
                        ema_results[f"EMA_{period}"] = {
                            "value": np.nan,
                            "data": [],
                            "period": period
                        }
                        continue
                    
                    # Initialize EMA with SMA for consistent behavior with original version
                    # First, calculate SMA for the initial period points
                    initial_sma = df['close'].iloc[:period].mean()
                    
                    # Then use pandas EWM function with SMA as the starting point
                    # Use the com parameter (center of mass) which is period-1 for equivalent alpha
                    alpha = 2 / (period + 1)
                    ema = df['close'].ewm(alpha=alpha, adjust=False).mean()
                    
                    # Store the result
                    ema_results[f"EMA_{period}"] = {
                        "value": safe_get_value(ema),
                        "data": ema.values.tolist() if len(ema) > 0 else [],
                        "period": period
                    }
                except Exception as e:
                    logger.error(f"Error calculating EMA_{period}: {str(e)}")
                    ema_results[f"EMA_{period}"] = {
                        "value": np.nan,
                        "data": [],
                        "period": period,
                        "error": str(e)
                    }
            
            # Add results to the main dictionary
            all_indicators["moving_averages"] = {
                "simple": sma_results,
                "exponential": ema_results
            }
            
            return True
        except Exception as e:
            logger.error(f"Error calculating moving_averages: {str(e)}")
            all_indicators["moving_averages"] = {
                "error": str(e),
                "simple": {},
                "exponential": {}
            }
            return False
    
    # Calculate Momentum Indicators
    def calculate_momentum():
        # RSI
        rsi = relative_strength_index(df['close'])
        rsi_results = {
            "RSI": {
                "value": safe_get_value(rsi),
                "data": rsi.values.tolist(),
                "overbought": 70,
                "oversold": 30
            }
        }
        
        # MACD
        macd_line, signal_line, histogram = macd(df['close'])
        macd_results = {
            "MACD": {
                "line": safe_get_value(macd_line),
                "signal": safe_get_value(signal_line),
                "histogram": safe_get_value(histogram),
                "line_data": macd_line.values.tolist(),
                "signal_data": signal_line.values.tolist(),
                "histogram_data": histogram.values.tolist()
            }
        }
        
        # Stochastic Oscillator
        k, d = stochastic_oscillator(df['high'], df['low'], df['close'])
        stoch_results = {
            "Stochastic": {
                "k": safe_get_value(k),
                "d": safe_get_value(d),
                "k_data": k.values.tolist(),
                "d_data": d.values.tolist(),
                "overbought": 80,
                "oversold": 20
            }
        }
        
        # CCI (Commodity Channel Index)
        cci = commodity_channel_index(df['high'], df['low'], df['close'])
        cci_results = {
            "CCI": {
                "value": safe_get_value(cci),
                "data": cci.values.tolist(),
                "overbought": 100,
                "oversold": -100
            }
        }
        
        # Williams %R - Using our implemented function
        williams_r = williams_percent_r(df['high'], df['low'], df['close'])
        williams_results = {
            "Williams_%R": {
                "value": safe_get_value(williams_r),
                "data": williams_r.values.tolist(),
                "overbought": -20,
                "oversold": -80
            }
        }
        
        # Rate of Change
        roc = rate_of_change(df['close'])
        roc_results = {
            "ROC": {
                "value": safe_get_value(roc),
                "data": roc.values.tolist()
            }
        }
        
        # Momentum (n-period price change)
        momentum = df['close'].diff(10)
        momentum_results = {
            "Momentum": {
                "value": safe_get_value(momentum),
                "data": momentum.values.tolist()
            }
        }
        
        # Awesome Oscillator
        ao = awesome_oscillator(df['high'], df['low'])
        ao_results = {
            "Awesome_Oscillator": {
                "value": safe_get_value(ao),
                "data": ao.values.tolist()
            }
        }
        
        # True Strength Index
        tsi = true_strength_index(df['close'])
        tsi_results = {
            "TSI": {
                "value": safe_get_value(tsi),
                "data": tsi.values.tolist(),
                "overbought": 25,
                "oversold": -25
            }
        }
        
        # Percentage Price Oscillator
        ppo, ppo_signal, ppo_hist = percentage_price_oscillator(df['close'])
        ppo_results = {
            "PPO": {
                "line": safe_get_value(ppo),
                "signal": safe_get_value(ppo_signal),
                "histogram": safe_get_value(ppo_hist),
                "line_data": ppo.values.tolist(),
                "signal_data": ppo_signal.values.tolist(),
                "histogram_data": ppo_hist.values.tolist()
            }
        }
        
        # Coppock Curve
        cc = coppock_curve(df['close'])
        cc_results = {
            "Coppock_Curve": {
                "value": safe_get_value(cc),
                "data": cc.values.tolist()
            }
        }
        
        # Combine all results
        return {
            **rsi_results, 
            **macd_results, 
            **stoch_results, 
            **cci_results, 
            **williams_results,
            **roc_results,
            **momentum_results,
            **ao_results,
            **tsi_results,
            **ppo_results,
            **cc_results
        }
    
    # Calculate Volume Indicators
    def calculate_volume_analysis():
        # On-Balance Volume
        obv = on_balance_volume(df['close'], df['volume'])
        obv_results = {
            "OBV": {
                "value": safe_get_value(obv),
                "data": obv.values.tolist()
            }
        }
        
        # Accumulation/Distribution Line
        adl = accumulation_distribution_line(df['high'], df['low'], df['close'], df['volume'])
        adl_results = {
            "ADL": {
                "value": safe_get_value(adl),
                "data": adl.values.tolist()
            }
        }
        
        # Chaikin Money Flow
        cmf = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        cmf_results = {
            "CMF": {
                "value": safe_get_value(cmf),
                "data": cmf.values.tolist()
            }
        }
        
        # Chaikin Oscillator
        co = chaikin_oscillator(df['high'], df['low'], df['close'], df['volume'])
        co_results = {
            "Chaikin_Oscillator": {
                "value": safe_get_value(co),
                "data": co.values.tolist()
            }
        }
        
        # Money Flow Index
        mfi = money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        mfi_results = {
            "MFI": {
                "value": safe_get_value(mfi),
                "data": mfi.values.tolist(),
                "overbought": 80,
                "oversold": 20
            }
        }
        
        # Volume-Weighted MACD (custom)
        # Use regular MACD but scale it by relative volume
        volume_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
        macd_line, signal_line, histogram = macd(df['close'])
        vol_macd = macd_line * volume_ratio
        vol_signal = signal_line.rolling(window=9).mean()
        vol_histogram = vol_macd - vol_signal
        vol_macd_results = {
            "Volume_MACD": {
                "line": safe_get_value(vol_macd),
                "signal": safe_get_value(vol_signal),
                "histogram": safe_get_value(vol_histogram),
                "line_data": vol_macd.values.tolist(),
                "signal_data": vol_signal.values.tolist(),
                "histogram_data": vol_histogram.values.tolist()
            }
        }
        
        # Volume Zone Oscillator
        vzo, vzo_signal = volume_zone_oscillator(df['close'], df['volume'])
        vzo_results = {
            "VZO": {
                "value": safe_get_value(vzo),
                "signal": safe_get_value(vzo_signal),
                "data": vzo.values.tolist(),
                "signal_data": vzo_signal.values.tolist()
            }
        }
        
        # Relative Volume
        rel_vol = relative_volume(df['volume'])
        rel_vol_results = {
            "Relative_Volume": {
                "value": safe_get_value(rel_vol),
                "data": rel_vol.values.tolist()
            }
        }
        
        # Combine all results
        return {
            **obv_results, 
            **adl_results, 
            **cmf_results, 
            **co_results,
            **mfi_results,
            **vol_macd_results,
            **vzo_results,
            **rel_vol_results
        }
    
    # Calculate Volatility Indicators
    def calculate_volatility():
        # Average True Range
        atr = average_true_range(df['high'], df['low'], df['close'])
        atr_results = {
            "ATR": {
                "value": safe_get_value(atr),
                "data": atr.values.tolist()
            }
        }
        
        # ATR Percent
        atr_percent = (atr / df['close']) * 100
        atr_percent_results = {
            "ATR_Percent": {
                "value": safe_get_value(atr_percent),
                "data": atr_percent.values.tolist()
            }
        }
        
        # Historical Volatility (close-to-close)
        returns = np.log(df['close'] / df['close'].shift(1))
        hist_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100
        hist_vol_results = {
            "Historical_Volatility": {
                "value": safe_get_value(hist_vol),
                "data": hist_vol.values.tolist()
            }
        }
        
        # Chandelier Exit
        long_exit, short_exit = chandelier_exit(df['high'], df['low'], df['close'])
        chandelier_results = {
            "Chandelier_Exit": {
                "long_exit": safe_get_value(long_exit),
                "short_exit": safe_get_value(short_exit),
                "long_exit_data": long_exit.values.tolist(),
                "short_exit_data": short_exit.values.tolist()
            }
        }
        
        # Parabolic SAR
        sar, trend = parabolic_sar(df['high'], df['low'])
        sar_results = {
            "Parabolic_SAR": {
                "value": safe_get_value(sar),
                "trend": safe_get_value(trend),
                "data": sar.values.tolist(),
                "trend_data": trend.values.tolist()
            }
        }
        
        # Bollinger Bandwidth
        upper, middle, lower = bollinger_bands(df['close'])
        bb_width = (upper - lower) / middle
        bb_width_results = {
            "BB_Width": {
                "value": safe_get_value(bb_width),
                "data": bb_width.values.tolist()
            }
        }
        
        # Combine all results
        return {
            **atr_results, 
            **atr_percent_results, 
            **hist_vol_results, 
            **chandelier_results,
            **sar_results,
            **bb_width_results
        }
    
    # Calculate Fibonacci Retracement Levels
    def calculate_fibonacci_levels():
        # Get recent high and low
        if len(df) < 20:
            return {}
            
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Calculate Fibonacci retracement levels
        diff = recent_high - recent_low
        fib_levels = {
            "Fibonacci_Levels": {
                "trend": "up" if df['close'].iloc[-1] > df['close'].iloc[-10] else "down",
                "high": float(recent_high),
                "low": float(recent_low),
                "0.0": float(recent_low),
                "0.236": float(recent_low + 0.236 * diff),
                "0.382": float(recent_low + 0.382 * diff),
                "0.5": float(recent_low + 0.5 * diff),
                "0.618": float(recent_low + 0.618 * diff),
                "0.786": float(recent_low + 0.786 * diff),
                "1.0": float(recent_high),
                "1.272": float(recent_high + 0.272 * diff),
                "1.618": float(recent_high + 0.618 * diff)
            }
        }
        
        return fib_levels
    
    # Calculate Technical Analysis Summary & Signals
    def calculate_market_structure():
        # Current price
        current_price = df['close'].iloc[-1]
        
        # Key Moving Averages
        sma50 = df['close'].rolling(window=50).mean().iloc[-1]
        sma200 = df['close'].rolling(window=200).mean().iloc[-1]
        
        # Trend Determination
        try:
            sma50_prev = df['close'].rolling(window=50).mean().iloc[-2]
            sma200_prev = df['close'].rolling(window=200).mean().iloc[-2]
            golden_cross = (sma50 > sma200) and (sma50_prev <= sma200_prev)
            death_cross = (sma50 < sma200) and (sma50_prev >= sma200_prev)
        except Exception as e:
            logger.error(f"Error calculating crosses: {str(e)}")
            golden_cross = False
            death_cross = False
        
        # Market Regime
        regime_series, volatility_series = market_regime(df['close'])
        
        # Extract scalar values using safe_get_value to prevent Series ambiguity
        regime_value = safe_get_value(regime_series, 0)
        volatility_flag = bool(safe_get_value(volatility_series, False))
        
        # Price relative to MAs
        above_50ma = current_price > sma50
        above_200ma = current_price > sma200
        
        # Pivot Points (traditional)
        pivot, r1, s1, r2, s2, r3, s3 = pivot_points(df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
        
        # Advanced Pivot Points
        if 'open' in df.columns:
            # Camarilla Pivot Points
            cam_pivot, cam_supports, cam_resistances = camarilla_pivot_points(df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
            
            # Woodie Pivot Points
            wood_pivot, wood_supports, wood_resistances = woodie_pivot_points(df['open'].iloc[-1], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
            
            # DeMark Pivot Points
            demark_pivot, demark_supports, demark_resistances = demark_pivot_points(df['open'].iloc[-1], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
        else:
            cam_pivot, cam_supports, cam_resistances = None, {}, {}
            wood_pivot, wood_supports, wood_resistances = None, {}, {}
            demark_pivot, demark_supports, demark_resistances = None, {}, {}
        
        # Volume Profile Analysis
        vp = volume_profile(df['high'], df['low'], df['close'], df['volume'], bins=10, window=30)
        
        # Harmonic Patterns
        hp = harmonic_patterns(df['high'], df['low'], df['close'])
        
        # Divergence with RSI
        rsi = relative_strength_index(df['close'])
        rsi_divergence = divergence_scanner(df['close'], rsi)
        
        # Extract scalar boolean values for divergences using safe_get_value
        # to handle cases where the divergence_scanner might return Series objects
        rsi_reg_bullish = safe_get_value(rsi_divergence["regular_bullish"], False)
        rsi_reg_bearish = safe_get_value(rsi_divergence["regular_bearish"], False)
        rsi_hid_bullish = safe_get_value(rsi_divergence["hidden_bullish"], False)
        rsi_hid_bearish = safe_get_value(rsi_divergence["hidden_bearish"], False)
        rsi_strength = safe_get_value(rsi_divergence["strength"], 0.0)
        
        # Divergence with MACD
        macd_line, _, _ = macd(df['close'])
        macd_divergence = divergence_scanner(df['close'], macd_line)
        
        # Extract scalar boolean values for MACD divergences
        macd_reg_bullish = safe_get_value(macd_divergence["regular_bullish"], False)
        macd_reg_bearish = safe_get_value(macd_divergence["regular_bearish"], False)
        macd_hid_bullish = safe_get_value(macd_divergence["hidden_bullish"], False)
        macd_hid_bearish = safe_get_value(macd_divergence["hidden_bearish"], False)
        macd_strength = safe_get_value(macd_divergence["strength"], 0.0)
        
        # Return comprehensive market structure analysis
        return {
            "Market_Structure": {
                "current_price": float(current_price),
                "trend": "bullish" if above_50ma and above_200ma else "bearish" if not above_50ma and not above_200ma else "mixed",
                "golden_cross": bool(golden_cross),
                "death_cross": bool(death_cross),
                "above_50ma": bool(above_50ma),
                "above_200ma": bool(above_200ma),
                "market_regime": float(regime_value),
                "high_volatility": bool(volatility_flag),
                
                # Standard Pivot Points
                "pivot": float(pivot),
                "r1": float(r1),
                "r2": float(r2),
                "r3": float(r3),
                "s1": float(s1),
                "s2": float(s2),
                "s3": float(s3),
                
                # Advanced Pivot Points
                "camarilla": {
                    "pivot": float(cam_pivot) if cam_pivot is not None else None,
                    "r1": float(cam_resistances.get('r1', 0)) if cam_resistances else None,
                    "r2": float(cam_resistances.get('r2', 0)) if cam_resistances else None,
                    "r3": float(cam_resistances.get('r3', 0)) if cam_resistances else None,
                    "r4": float(cam_resistances.get('r4', 0)) if cam_resistances else None,
                    "s1": float(cam_supports.get('s1', 0)) if cam_supports else None,
                    "s2": float(cam_supports.get('s2', 0)) if cam_supports else None,
                    "s3": float(cam_supports.get('s3', 0)) if cam_supports else None,
                    "s4": float(cam_supports.get('s4', 0)) if cam_supports else None
                },
                "woodie": {
                    "pivot": float(wood_pivot) if wood_pivot is not None else None,
                    "r1": float(wood_resistances.get('r1', 0)) if wood_resistances else None,
                    "r2": float(wood_resistances.get('r2', 0)) if wood_resistances else None,
                    "s1": float(wood_supports.get('s1', 0)) if wood_supports else None,
                    "s2": float(wood_supports.get('s2', 0)) if wood_supports else None
                },
                "demark": {
                    "pivot": float(demark_pivot) if demark_pivot is not None else None,
                    "r1": float(demark_resistances.get('r1', 0)) if demark_resistances else None,
                    "s1": float(demark_supports.get('s1', 0)) if demark_supports else None
                },
                
                # Volume Profile
                "volume_profile": {
                    "poc": float(vp["poc"]) if vp["poc"] is not None else None,
                    "vah": float(vp["vah"]) if vp["vah"] is not None else None,
                    "val": float(vp["val"]) if vp["val"] is not None else None,
                    "hvn_count": len(vp["hvn"]),
                    "lvn_count": len(vp["lvn"]),
                    "hvn_prices": [float(price) for price in vp["hvn"][:3]] if vp["hvn"] else []  # Include top 3 HVNs
                },
                
                # Harmonic Patterns
                "harmonic_patterns": {
                    "patterns_found": len(hp["patterns"]),
                    "active_patterns": [
                        {
                            "type": pattern["type"],
                            "confidence": pattern["confidence"],
                            "completion_price": float(pattern["completion_price"])
                        }
                        for pattern in hp["patterns"][:2]  # Include top 2 patterns
                    ] if hp["patterns"] else []
                },
                
                # Divergences - Using extracted scalar values
                "divergences": {
                    "rsi": {
                        "regular_bullish": bool(rsi_reg_bullish),
                        "regular_bearish": bool(rsi_reg_bearish),
                        "hidden_bullish": bool(rsi_hid_bullish),
                        "hidden_bearish": bool(rsi_hid_bearish),
                        "strength": float(rsi_strength)
                    },
                    "macd": {
                        "regular_bullish": bool(macd_reg_bullish),
                        "regular_bearish": bool(macd_reg_bearish),
                        "hidden_bullish": bool(macd_hid_bullish),
                        "hidden_bearish": bool(macd_hid_bearish),
                        "strength": float(macd_strength)
                    }
                }
            }
        }
    
    # Calculate multi-timeframe analysis if weekly data is available
    def calculate_multi_timeframe_analysis():
        try:
            # Check if we have enough data for daily timeframe
            if len(df) < 50:
                return {}
                
            # Create weekly data by resampling
            # This is a simple approximation of weekly data for demonstration
            # In production, you'd want to pass actual weekly data
            weekly_df = df.copy()
            weekly_df.index = pd.to_datetime(weekly_df.index) if not isinstance(weekly_df.index, pd.DatetimeIndex) else weekly_df.index
            weekly_df = weekly_df.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Only calculate if we have enough weekly data
            if len(weekly_df) < 15:
                return {}
                
            # Calculate Elder Triple Screen components
            elder_results = elder_triple_screen(
                df['close'], df['high'], df['low'], df['volume'],
                weekly_df['close'], weekly_df['high'], weekly_df['low'], weekly_df['volume']
            )
            
            # Format results
            multi_tf_results = {
                "Elder_Triple_Screen": {
                    "weekly_trend": int(elder_results['weekly_trend']),
                    "weekly_rsi": float(safe_get_value(elder_results['weekly_rsi'])),
                    "impulse": int(safe_get_value(elder_results['impulse'])),
                    "buy_signal": bool(safe_get_value(elder_results['buy_signal'])),
                    "sell_signal": bool(safe_get_value(elder_results['sell_signal'])),
                    "force_index": float(safe_get_value(elder_results['force_index'])),
                    "impulse_data": elder_results['impulse'].values.tolist() if isinstance(elder_results['impulse'], pd.Series) else [],
                    "weekly_rsi_data": elder_results['weekly_rsi'].values.tolist() if isinstance(elder_results['weekly_rsi'], pd.Series) else [],
                    "force_index_data": elder_results['force_index'].values.tolist() if isinstance(elder_results['force_index'], pd.Series) else []
                }
            }
            
            return multi_tf_results
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe analysis: {str(e)}")
            return {
                "Elder_Triple_Screen": {
                    "error": str(e)
                }
            }
    
    # Calculate Additional Technical Indicators that weren't included above
    def calculate_additional_indicators():
        try:
            # Keltner Channel
            upper_kc, middle_kc, lower_kc = keltner_channel(df['high'], df['low'], df['close'])
            keltner_results = {
                "KC_Upper": {
                    "value": safe_get_value(upper_kc),
                    "data": upper_kc.values.tolist()
                },
                "KC_Middle": {
                    "value": safe_get_value(middle_kc),
                    "data": middle_kc.values.tolist()
                },
                "KC_Lower": {
                    "value": safe_get_value(lower_kc),
                    "data": lower_kc.values.tolist()
                }
            }
            
            # Elder Ray Index
            bull_power, bear_power = elder_ray_index(df['high'], df['low'], df['close'])
            elder_ray_results = {
                "Bull_Power": {
                    "value": safe_get_value(bull_power),
                    "data": bull_power.values.tolist()
                },
                "Bear_Power": {
                    "value": safe_get_value(bear_power),
                    "data": bear_power.values.tolist()
                }
            }
            
            # Klinger Oscillator
            kvo, kvo_signal = klinger_oscillator(df['high'], df['low'], df['close'], df['volume'])
            klinger_results = {
                "Klinger_Osc": {
                    "value": safe_get_value(kvo),
                    "data": kvo.values.tolist()
                },
                "Klinger_Signal": {
                    "value": safe_get_value(kvo_signal),
                    "data": kvo_signal.values.tolist()
                }
            }
            
            # Aroon Indicator
            aroon_up, aroon_down = aroon_indicator(df['high'], df['low'])
            aroon_results = {
                "Aroon_Up": {
                    "value": safe_get_value(aroon_up),
                    "data": aroon_up.values.tolist()
                },
                "Aroon_Down": {
                    "value": safe_get_value(aroon_down),
                    "data": aroon_down.values.tolist()
                }
            }
            
            # Donchian Channels
            upper_dc, middle_dc, lower_dc = donchian_channels(df['high'], df['low'])
            donchian_results = {
                "DC_Upper": {
                    "value": safe_get_value(upper_dc),
                    "data": upper_dc.values.tolist()
                },
                "DC_Middle": {
                    "value": safe_get_value(middle_dc),
                    "data": middle_dc.values.tolist()
                },
                "DC_Lower": {
                    "value": safe_get_value(lower_dc),
                    "data": lower_dc.values.tolist()
                }
            }
            
            # Vortex Indicator
            vortex_pos, vortex_neg = vortex_indicator(df['high'], df['low'], df['close'])
            vortex_results = {
                "VI_Plus": {
                    "value": safe_get_value(vortex_pos),
                    "data": vortex_pos.values.tolist()
                },
                "VI_Minus": {
                    "value": safe_get_value(vortex_neg),
                    "data": vortex_neg.values.tolist()
                }
            }
            
            # Mass Index
            mass_idx = mass_index(df['high'], df['low'])
            mass_idx_results = {
                "Mass_Index": {
                    "value": safe_get_value(mass_idx),
                    "data": mass_idx.values.tolist()
                }
            }
            
            # Hull Moving Average
            hma = hull_moving_average(df['close'])
            hma_results = {
                "HMA": {
                    "value": safe_get_value(hma),
                    "data": hma.values.tolist()
                }
            }
            
            # VWAP (Volume-Weighted Average Price)
            # Using our own implementation since vwap is not defined
            def calculate_vwap(high, low, close, volume):
                typical_price = (high + low + close) / 3
                vwap = (typical_price * volume).cumsum() / volume.cumsum()
                return vwap
                
            vwap_values = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
            vwap_results = {
                "VWAP": {
                    "value": safe_get_value(vwap_values),
                    "data": vwap_values.values.tolist()
                }
            }
            
            # Ichimoku Cloud
            tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ichimoku_cloud(df['high'], df['low'], df['close'])
            
            # Extract the latest values for comparisons
            latest_senkou_a = safe_get_value(senkou_span_a)
            latest_senkou_b = safe_get_value(senkou_span_b)
            latest_close = df['close'].iloc[-1]
            
            # Compare scalar values instead of Series
            is_bullish = latest_senkou_a > latest_senkou_b if not pd.isna(latest_senkou_a) and not pd.isna(latest_senkou_b) else False
            is_price_above_cloud = (latest_close > latest_senkou_a and latest_close > latest_senkou_b) if not pd.isna(latest_senkou_a) and not pd.isna(latest_senkou_b) else False
            
            ichimoku_results = {
                "Ichimoku": {
                    "tenkan_sen": safe_get_value(tenkan_sen),
                    "kijun_sen": safe_get_value(kijun_sen),
                    "senkou_span_a": latest_senkou_a,
                    "senkou_span_b": latest_senkou_b,
                    "chikou_span": safe_get_value(chikou_span),
                    "tenkan_data": tenkan_sen.values.tolist(),
                    "kijun_data": kijun_sen.values.tolist(),
                    "senkou_a_data": senkou_span_a.values.tolist(),
                    "senkou_b_data": senkou_span_b.values.tolist(),
                    "chikou_data": chikou_span.values.tolist(),
                    "bullish": bool(is_bullish),
                    "price_above_cloud": bool(is_price_above_cloud)
                }
            }
            
            # Supertrend indicator
            supertrend_values, direction_values = supertrend(df['high'], 
                                                             df['low'], 
                                                             df['close'], 
                                                             period=10, 
                                                             multiplier=3.0)
            
            # Extract the latest values for comparison
            latest_direction = safe_get_value(direction_values, 0)
            previous_direction = safe_get_value(direction_values.shift(1), 0) if len(direction_values) > 1 else 0
            
            # Determine signal based on scalar values
            if latest_direction == 1 and previous_direction != 1:
                signal = 'buy'
            elif latest_direction == -1 and previous_direction != -1:
                signal = 'sell'
            else:
                signal = 'hold'
                
            supertrend_results = {
                'value': safe_get_value(supertrend_values),
                'direction': latest_direction,
                'signal': signal
            }
            
            # Heikin-Ashi
            if 'open' in df.columns:
                ha_open, ha_high, ha_low, ha_close = heikin_ashi(df['open'], df['high'], df['low'], df['close'])
                heikin_ashi_results = {
                    "Heikin_Ashi": {
                        "open": safe_get_value(ha_open),
                        "high": safe_get_value(ha_high),
                        "low": safe_get_value(ha_low),
                        "close": safe_get_value(ha_close),
                        "open_data": ha_open.values.tolist(),
                        "high_data": ha_high.values.tolist(),
                        "low_data": ha_low.values.tolist(),
                        "close_data": ha_close.values.tolist(),
                        "trend": "bullish" if safe_get_value(ha_close) > safe_get_value(ha_open) else "bearish"
                    }
                }
            else:
                heikin_ashi_results = {}
            
            # Squeeze Momentum
            try:
                squeeze_mom, squeeze_on = squeeze_momentum(df['high'], df['low'], df['close'])
                squeeze_results = {
                    "Squeeze_Momentum": {
                        "momentum": safe_get_value(squeeze_mom),
                        "squeeze_on": safe_get_value(squeeze_on),
                        "momentum_data": squeeze_mom.values.tolist(),
                        "squeeze_on_data": squeeze_on.values.tolist(),
                        "momentum_increasing": safe_get_value(squeeze_mom) > safe_get_value(squeeze_mom.shift(1)) if not pd.isna(safe_get_value(squeeze_mom.shift(1))) else False
                    }
                }
            except Exception as e:
                logger.error(f"Error calculating Squeeze Momentum: {str(e)}")
                squeeze_results = {"Squeeze_Momentum": {"error": str(e)}}
            
            # Ehlers Fisher Transform
            fisher, fisher_signal = ehlers_fisher_transform(df['close'])
            fisher_results = {
                "Fisher_Transform": {
                    "value": safe_get_value(fisher),
                    "signal": safe_get_value(fisher_signal),
                    "data": fisher.values.tolist(),
                    "signal_data": fisher_signal.values.tolist(),
                    "bullish": safe_get_value(fisher) > safe_get_value(fisher_signal) if not pd.isna(safe_get_value(fisher_signal)) else False,
                    "overbought": safe_get_value(fisher) > 2.0,
                    "oversold": safe_get_value(fisher) < -2.0
                }
            }
            
            # Chande Momentum Oscillator
            cmo = chande_momentum_oscillator(df['close'])
            cmo_results = {
                "CMO": {
                    "value": safe_get_value(cmo),
                    "data": cmo.values.tolist(),
                    "overbought": 50,
                    "oversold": -50
                }
            }
            
            # Stochastic RSI
            k, d = stochastic_rsi(df['close'], 
                                                 rsi_period=14, 
                                                 stoch_period=14, 
                                                 k_period=3, 
                                                 d_period=3)
            stochastic_rsi_results = {
                'k': safe_get_value(k),
                'd': safe_get_value(d),
                'overbought': k.iloc[-1] > 80 if not pd.isna(k.iloc[-1]) else None,
                'oversold': k.iloc[-1] < 20 if not pd.isna(k.iloc[-1]) else None,
                'signal': 'bullish' if k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2] else
                         'bearish' if k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2] else 'neutral'
            }
            
            # Mean Reversion Index
            mri, upper, lower = mean_reversion_index(df['close'], 
                                                                      df['high'], 
                                                                      df['low'], 
                                                                      period=14)
            mean_reversion_index_results = {
                'value': safe_get_value(mri),
                'upper_band': safe_get_value(upper),
                'lower_band': safe_get_value(lower),
                'signal': 'buy' if mri.iloc[-1] > 50 else 'sell' if mri.iloc[-1] < -50 else 'neutral',
                'strength': abs(mri.iloc[-1]) / 100 if not pd.isna(mri.iloc[-1]) else None
            }
            
            # Combine all results
            return {
                **keltner_results,
                **elder_ray_results,
                **klinger_results,
                **aroon_results,
                **donchian_results,
                **vortex_results,
                **mass_idx_results,
                **hma_results,
                **vwap_results,
                **ichimoku_results,
                **supertrend_results,
                **heikin_ashi_results,
                **squeeze_results,
                **fisher_results,
                **cmo_results,
                **stochastic_rsi_results,
                **mean_reversion_index_results
            }
        except Exception as e:
            logger.error(f"Error calculating additional indicators: {str(e)}")
            return {
                "error": f"Error calculating additional indicators: {str(e)}"
            }

    # Add this new function after calculate_additional_indicators()
    def calculate_advanced_patterns():
        try:
            advanced_patterns_result = {}  # Initialize a local result dictionary
            
            # Elliott Wave Pattern Detection
            if len(df) >= 50:  # Need sufficient data for pattern detection
                try:
                    elliott_waves = elliott_wave_tracker(df['high'], 
                                                df['low'], 
                                                df['close'], 
                                                df['volume'])
                    advanced_patterns_result['elliott_wave'] = {
                        'wave_count': elliott_waves['wave_count'],
                        'current_position': elliott_waves['current_position'],
                        'confidence': elliott_waves['confidence'],
                        'impulse_waves': elliott_waves['impulse_waves'],
                        'corrective_waves': elliott_waves['corrective_waves']
                    }
                except Exception as e:
                    logger.error(f"Error calculating Elliott Wave: {str(e)}")
                    advanced_patterns_result['elliott_wave'] = {"error": str(e)}
            
            # Divergence Scanning between price and RSI
            if len(df) >= 20:
                try:
                    rsi = relative_strength_index(df['close'])
                    divergences = divergence_scanner(df['close'], rsi)
                    advanced_patterns_result['divergences'] = {
                        'regular_bullish': divergences['regular_bullish'],
                        'regular_bearish': divergences['regular_bearish'],
                        'hidden_bullish': divergences['hidden_bullish'],
                        'hidden_bearish': divergences['hidden_bearish'],
                        'strength': divergences['strength']
                    }
                except Exception as e:
                    logger.error(f"Error calculating Divergences: {str(e)}")
                    advanced_patterns_result['divergences'] = {"error": str(e)}
            
            # Harmonic Pattern Detection
            if len(df) >= 30:
                try:
                    harmonic_patterns_result = harmonic_patterns(df['high'], df['low'], df['close'])
                    advanced_patterns_result['harmonic_patterns'] = harmonic_patterns_result['patterns']
                except Exception as e:
                    logger.error(f"Error calculating Harmonic Patterns: {str(e)}")
                    advanced_patterns_result['harmonic_patterns'] = {"error": str(e)}
            
            # Volume Profile Analysis
            if len(df) >= 30:
                try:
                    vol_profile = volume_profile(df['high'], df['low'], df['close'], df['volume'])
                    advanced_patterns_result['volume_profile'] = {
                        'poc': vol_profile['poc'],  # Point of Control
                        'vah': vol_profile['vah'],  # Value Area High
                        'val': vol_profile['val'],  # Value Area Low
                        'hvn': vol_profile['hvn'],  # High Volume Nodes
                        'lvn': vol_profile['lvn']   # Low Volume Nodes
                    }
                except Exception as e:
                    logger.error(f"Error calculating Volume Profile: {str(e)}")
                    advanced_patterns_result['volume_profile'] = {"error": str(e)}
            
            # Heikin-Ashi Analysis for trend smoothing
            if len(df) >= 10 and 'open' in df.columns:
                try:
                    ha_open, ha_high, ha_low, ha_close = heikin_ashi(
                        df['open'], df['high'], df['low'], df['close'])
                    
                    # Determine trend based on Heikin-Ashi
                    ha_trend = "uptrend"
                    if ha_close.iloc[-1] < ha_open.iloc[-1] and ha_close.iloc[-2] < ha_open.iloc[-2]:
                        ha_trend = "downtrend"
                    elif ha_close.iloc[-1] > ha_open.iloc[-1] and ha_close.iloc[-2] < ha_open.iloc[-2]:
                        ha_trend = "reversal_up"
                    elif ha_close.iloc[-1] < ha_open.iloc[-1] and ha_close.iloc[-2] > ha_open.iloc[-2]:
                        ha_trend = "reversal_down"
                    
                    advanced_patterns_result['heikin_ashi'] = {
                        'trend': ha_trend,
                        'open': safe_get_value(ha_open),
                        'high': safe_get_value(ha_high),
                        'low': safe_get_value(ha_low),
                        'close': safe_get_value(ha_close)
                    }
                except Exception as e:
                    logger.error(f"Error calculating Heikin-Ashi: {str(e)}")
                    advanced_patterns_result['heikin_ashi'] = {"error": str(e)}
                
            # If we have bid and ask volume data, perform order flow analysis
            if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                try:
                    order_flow = orderflow_analysis(
                        df['close'], df['volume'], df['bid_volume'], df['ask_volume'])
                    
                    advanced_patterns_result['order_flow'] = {
                        'delta': safe_get_value(order_flow['delta']),
                        'cumulative_delta': safe_get_value(order_flow['cumulative_delta']),
                        'significant_buying': bool(order_flow['significant_buying'].iloc[-1]) if len(order_flow['significant_buying']) > 0 else None,
                        'significant_selling': bool(order_flow['significant_selling'].iloc[-1]) if len(order_flow['significant_selling']) > 0 else None,
                        'imbalance_ratio': safe_get_value(order_flow['imbalance_ratio']),
                        'absorption_zones': bool(order_flow['absorption_zones'].iloc[-1]) if len(order_flow['absorption_zones']) > 0 else None
                    }
                except Exception as e:
                    logger.error(f"Error calculating Order Flow Analysis: {str(e)}")
                    advanced_patterns_result['order_flow'] = {"error": str(e)}
            
            # Return the results to be merged with the main results
            return advanced_patterns_result
            
        except Exception as e:
            logger.error(f"Error calculating advanced patterns: {str(e)}")
            return {"error": str(e)}

    # Execute all indicator calculations with error handling
    try:
        # Track execution and errors for each calculation
        execution_results = {
            "moving_averages": {"executed": False, "error": None},
            "momentum": {"executed": False, "error": None},
            "volume_analysis": {"executed": False, "error": None},
            "volatility": {"executed": False, "error": None},
            "fibonacci_levels": {"executed": False, "error": None},
            "market_structure": {"executed": False, "error": None},
            "multi_timeframe_analysis": {"executed": False, "error": None},
            "additional_indicators": {"executed": False, "error": None},
            "advanced_patterns": {"executed": False, "error": None}
        }
        
        # Execute each function with individual try-except blocks
        try:
            calculate_moving_averages()
            execution_results["moving_averages"]["executed"] = True
        except Exception as e:
            execution_results["moving_averages"]["error"] = str(e)
            logger.error(f"Error calculating moving averages: {str(e)}")
            
        try:
            momentum_results = calculate_momentum()
            if momentum_results and isinstance(momentum_results, dict):
                result.update(momentum_results)
            execution_results["momentum"]["executed"] = True
        except Exception as e:
            execution_results["momentum"]["error"] = str(e)
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            
        try:
            volume_results = calculate_volume_analysis()
            if volume_results and isinstance(volume_results, dict):
                result.update(volume_results)
            execution_results["volume_analysis"]["executed"] = True
        except Exception as e:
            execution_results["volume_analysis"]["error"] = str(e)
            logger.error(f"Error calculating volume analysis: {str(e)}")
            
        try:
            volatility_results = calculate_volatility()
            if volatility_results and isinstance(volatility_results, dict):
                result.update(volatility_results)
            execution_results["volatility"]["executed"] = True
        except Exception as e:
            execution_results["volatility"]["error"] = str(e)
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            
        try:
            fib_results = calculate_fibonacci_levels()
            if fib_results and isinstance(fib_results, dict):
                result.update(fib_results)
            execution_results["fibonacci_levels"]["executed"] = True
        except Exception as e:
            execution_results["fibonacci_levels"]["error"] = str(e)
            logger.error(f"Error calculating fibonacci levels: {str(e)}")
            
        try:
            market_structure_results = calculate_market_structure()
            if market_structure_results and isinstance(market_structure_results, dict):
                result.update(market_structure_results)
            execution_results["market_structure"]["executed"] = True
        except Exception as e:
            execution_results["market_structure"]["error"] = str(e)
            logger.error(f"Error calculating market structure: {str(e)}")
            
        try:
            mtf_results = calculate_multi_timeframe_analysis()
            if mtf_results and isinstance(mtf_results, dict):
                result.update(mtf_results)
            execution_results["multi_timeframe_analysis"]["executed"] = True
        except Exception as e:
            execution_results["multi_timeframe_analysis"]["error"] = str(e)
            logger.error(f"Error calculating multi-timeframe analysis: {str(e)}")
            
        # Call additional indicators function
        try:
            add_results = calculate_additional_indicators()
            if add_results and isinstance(add_results, dict):
                result.update(add_results)
            execution_results["additional_indicators"]["executed"] = True
        except Exception as e:
            execution_results["additional_indicators"]["error"] = str(e)
            logger.error(f"Error calculating additional indicators: {str(e)}")
            
        # Call patterns function after calculating all other indicators
        try:
            adv_patterns_results = calculate_advanced_patterns()
            if adv_patterns_results and isinstance(adv_patterns_results, dict):
                result.update(adv_patterns_results)
            execution_results["advanced_patterns"]["executed"] = True
        except Exception as e:
            execution_results["advanced_patterns"]["error"] = str(e)
            logger.error(f"Error calculating advanced patterns: {str(e)}")
        
        # Add execution results to the output
        result["_execution_info"] = {
            "success_count": sum(1 for r in execution_results.values() if r["executed"]),
            "error_count": sum(1 for r in execution_results.values() if r["error"] is not None),
            "details": execution_results
        }
        
        # Return the collected indicators
        return result
    except Exception as e:
        logger.error(f"Error in technical indicator calculation: {str(e)}")
        # Return partial results if available, otherwise empty dict
        return result if result else {}

def summarize_pair_data(pair_data):
    """
    Create a concise summary of pair data for LLM analysis
    """
    if not pair_data or 'candles' not in pair_data or not pair_data['candles']:
        return "Insufficient data for pair analysis."
    
    # Convert candles to DataFrame
    candles = pd.DataFrame(pair_data['candles'])
    
    # Calculate basic statistics
    if len(candles) < 2:
        return "Insufficient candle data for technical analysis."
    
    # Basic statistics
    current_price = candles['close'].iloc[-1]
    prev_day_close = candles['close'].iloc[-2]
    price_change = current_price - prev_day_close
    price_change_pct = (price_change / prev_day_close) * 100
    
    # Volume analysis
    current_volume = candles['volume'].iloc[-1]
    avg_volume = candles['volume'].rolling(window=20).mean().iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    
    # Get order book data if available
    book_data = ""
    if 'order_book' in pair_data and pair_data['order_book']:
        bids = pair_data['order_book'].get('bids', [])
        asks = pair_data['order_book'].get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0]['price']) if bids else None
            best_ask = float(asks[0]['price']) if asks else None
            spread = best_ask - best_bid if best_bid and best_ask else None
            spread_pct = (spread / best_bid) * 100 if best_bid and spread else None
            
            book_data = f"""
Order Book Analysis:
- Best Bid: {best_bid}
- Best Ask: {best_ask}
- Spread: {spread} ({spread_pct:.4f}%)
- Total Bid Depth (Top 5): {sum(float(bid['size']) for bid in bids[:5]) if len(bids) >= 5 else 'N/A'}
- Total Ask Depth (Top 5): {sum(float(ask['size']) for ask in asks[:5]) if len(asks) >= 5 else 'N/A'}
"""
    
    # Get technical indicators
    indicators = calculate_technical_indicators(pair_data)
    
    # Prepare technical indicators summary
    indicators_summary = ""
    if indicators:
        # Moving Averages
        ma_summary = "Moving Averages:\n"
        for ma_type in ["SMA_20", "SMA_50", "SMA_200", "EMA_20"]:
            if ma_type in indicators:
                ma_val = indicators[ma_type]["value"]
                ma_summary += f"- {ma_type}: {ma_val}\n"
        
        # RSI
        rsi_indicator = indicators.get("RSI", {})
        rsi_val = "N/A"
        
        # Handle both formats: RSI as a direct float value or as a dictionary with a "value" key
        if isinstance(rsi_indicator, (int, float)):
            rsi_val = rsi_indicator
        elif isinstance(rsi_indicator, dict) and "value" in rsi_indicator:
            rsi_val = rsi_indicator["value"]
            
        rsi_status = "N/A"
        if isinstance(rsi_val, (int, float)):
            if rsi_val > 70:
                rsi_status = "Overbought"
            elif rsi_val < 30:
                rsi_status = "Oversold"
            else:
                rsi_status = "Neutral"
        
        # MACD
        macd_info = indicators.get("MACD", {})
        
        # Handle different possible formats for MACD
        if isinstance(macd_info, dict):
            macd_line = macd_info.get("line", "N/A")
            macd_signal = macd_info.get("signal", "N/A")
            macd_histogram = macd_info.get("histogram", "N/A")
        else:
            # If MACD is not a dictionary, set defaults
            macd_line = "N/A"
            macd_signal = "N/A"
            macd_histogram = "N/A"
            # Log for debugging
            logging.warning(f"MACD is not in expected format: {type(macd_info)}")
        
        # Bollinger Bands
        bb_info = indicators.get("Bollinger_Bands", {})
        
        # Handle different possible formats for Bollinger Bands
        if isinstance(bb_info, dict):
            bb_upper = bb_info.get("upper", "N/A")
            bb_lower = bb_info.get("lower", "N/A")
        else:
            # If Bollinger_Bands is not a dictionary, set defaults
            bb_upper = "N/A"
            bb_lower = "N/A"
            # Log for debugging
            logging.warning(f"Bollinger_Bands is not in expected format: {type(bb_info)}")
        
        # Market structure
        market_structure = indicators.get("Market_Structure", {})
        
        # Handle different possible formats for Market Structure
        if isinstance(market_structure, dict):
            trend_direction = market_structure.get("trend_direction", "N/A")
        else:
            # If Market_Structure is not a dictionary, set default trend direction
            trend_direction = "N/A"
            # Log for debugging
            logging.warning(f"Market_Structure is not in expected format: {type(market_structure)}")
        
        # Average True Range (ATR)
        atr_indicator = indicators.get("ATR", {})
        atr_val = "N/A"
        
        # Handle both formats: ATR as a direct float value or as a dictionary with a "value" key
        if isinstance(atr_indicator, (int, float)):
            atr_val = atr_indicator
        elif isinstance(atr_indicator, dict) and "value" in atr_indicator:
            atr_val = atr_indicator["value"]
        
        # Compile indicators summary
        indicators_summary = f"""
Technical Indicators:
{ma_summary}
- RSI: {rsi_val} ({rsi_status})
- MACD Line: {macd_line}
- MACD Signal: {macd_signal}
- MACD Histogram: {macd_histogram}
- Bollinger Upper: {bb_upper}
- Bollinger Lower: {bb_lower}
- ATR: {atr_val}
- Overall Trend: {trend_direction}
"""
    
    # Main summary
    summary = f"""
Market Analysis Summary for {pair_data.get('product_details', {}).get('product_id', 'Unknown Pair')}:

Price Information:
- Current Price: {current_price}
- Previous Close: {prev_day_close}
- Change: {price_change} ({price_change_pct:.2f}%)
- 24h High: {candles['high'].max()}
- 24h Low: {candles['low'].min()}

Volume Analysis:
- Current Volume: {current_volume}
- Average Volume (20-period): {avg_volume}
- Volume Ratio: {volume_ratio:.2f}x average

{book_data}
{indicators_summary}
"""
    
    return summary 