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
    coppock_curve, vwap, klinger_oscillator, ichimoku_cloud, supertrend, heikin_ashi,
    camarilla_pivot_points, woodie_pivot_points, demark_pivot_points, squeeze_momentum,
    ehlers_fisher_transform, chande_momentum_oscillator, elder_triple_screen
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
    
    # Create a dictionary to store all indicator results
    all_indicators = {}
    
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
        golden_cross = sma50 > sma200 and df['close'].rolling(window=50).mean().iloc[-2] <= df['close'].rolling(window=200).mean().iloc[-2]
        death_cross = sma50 < sma200 and df['close'].rolling(window=50).mean().iloc[-2] >= df['close'].rolling(window=200).mean().iloc[-2]
        
        # Market Regime
        regime, volatility_flag = market_regime(df['close'])
        
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
        
        # Divergence with MACD
        macd_line, _, _ = macd(df['close'])
        macd_divergence = divergence_scanner(df['close'], macd_line)
        
        # Return comprehensive market structure analysis
        return {
            "Market_Structure": {
                "current_price": float(current_price),
                "trend": "bullish" if above_50ma and above_200ma else "bearish" if not above_50ma and not above_200ma else "mixed",
                "golden_cross": bool(golden_cross),
                "death_cross": bool(death_cross),
                "above_50ma": bool(above_50ma),
                "above_200ma": bool(above_200ma),
                "market_regime": float(safe_get_value(regime, 0)),
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
                
                # Divergences
                "divergences": {
                    "rsi": {
                        "regular_bullish": bool(rsi_divergence["regular_bullish"]),
                        "regular_bearish": bool(rsi_divergence["regular_bearish"]),
                        "hidden_bullish": bool(rsi_divergence["hidden_bullish"]),
                        "hidden_bearish": bool(rsi_divergence["hidden_bearish"]),
                        "strength": float(rsi_divergence["strength"])
                    },
                    "macd": {
                        "regular_bullish": bool(macd_divergence["regular_bullish"]),
                        "regular_bearish": bool(macd_divergence["regular_bearish"]),
                        "hidden_bullish": bool(macd_divergence["hidden_bullish"]),
                        "hidden_bearish": bool(macd_divergence["hidden_bearish"]),
                        "strength": float(macd_divergence["strength"])
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
            vwap_values = vwap(df['high'], df['low'], df['close'], df['volume'])
            vwap_results = {
                "VWAP": {
                    "value": safe_get_value(vwap_values),
                    "data": vwap_values.values.tolist()
                }
            }
            
            # Ichimoku Cloud
            tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ichimoku_cloud(df['high'], df['low'], df['close'])
            ichimoku_results = {
                "Ichimoku": {
                    "tenkan_sen": safe_get_value(tenkan_sen),
                    "kijun_sen": safe_get_value(kijun_sen),
                    "senkou_span_a": safe_get_value(senkou_span_a),
                    "senkou_span_b": safe_get_value(senkou_span_b),
                    "chikou_span": safe_get_value(chikou_span),
                    "tenkan_data": tenkan_sen.values.tolist(),
                    "kijun_data": kijun_sen.values.tolist(),
                    "senkou_a_data": senkou_span_a.values.tolist(),
                    "senkou_b_data": senkou_span_b.values.tolist(),
                    "chikou_data": chikou_span.values.tolist(),
                    "bullish": safe_get_value(senkou_span_a > senkou_span_b),
                    "price_above_cloud": safe_get_value(df['close'] > senkou_span_a) and safe_get_value(df['close'] > senkou_span_b)
                }
            }
            
            # Supertrend
            supertrend_line, supertrend_direction = supertrend(df['high'], df['low'], df['close'])
            supertrend_results = {
                "Supertrend": {
                    "value": safe_get_value(supertrend_line),
                    "direction": safe_get_value(supertrend_direction),
                    "data": supertrend_line.values.tolist(),
                    "direction_data": supertrend_direction.values.tolist(),
                    "bullish": safe_get_value(supertrend_direction) == 1
                }
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
                **cmo_results
            }
        except Exception as e:
            logger.error(f"Error calculating additional indicators: {str(e)}")
            return {
                "error": f"Error calculating additional indicators: {str(e)}"
            }

    # Execute all indicator calculations with error handling
    try:
        # Create a dictionary to track execution results
        execution_results = {}
        
        # Calculate basic indicators first
        # RSI
        rsi = relative_strength_index(df['close'])
        all_indicators["RSI"] = {
            "value": safe_get_value(rsi),
            "data": rsi.values.tolist(),
            "overbought": 70,
            "oversold": 30
        }
        
        # Call each calculation function with error handling
        for func_name, func in [
            ("moving_averages", calculate_moving_averages),
            ("momentum_indicators", calculate_momentum),
            ("volume_indicators", calculate_volume_analysis),
            ("volatility_indicators", calculate_volatility),
            ("fibonacci_levels", calculate_fibonacci_levels),
            ("market_structure", calculate_market_structure),
            ("multi_timeframe", calculate_multi_timeframe_analysis),
            ("additional_indicators", calculate_additional_indicators)
        ]:
            try:
                result = func()
                execution_results[func_name] = result
                
                # If this is market_structure, ensure it's added directly to all_indicators
                if func_name == "market_structure" and result and isinstance(result, dict):
                    # Add all values from the result dictionary to all_indicators
                    for key, value in result.items():
                        all_indicators[key] = value
                    
                    # Log the keys so we can debug what's available
                    logger.info(f"Market structure keys added to indicators: {list(result.keys())}")
                
                # For momentum_indicators, volume_indicators, volatility_indicators, and additional_indicators
                # Make sure their results are directly added to all_indicators
                elif func_name in ["momentum_indicators", "volume_indicators", "volatility_indicators", "additional_indicators"] and result:
                    if isinstance(result, dict):
                        # Add all values from the result dictionary to all_indicators
                        for key, value in result.items():
                            all_indicators[key] = value
                        
                        # Log the keys for debugging
                        logger.info(f"{func_name} keys added to indicators: {list(result.keys())}")
            except Exception as e:
                logger.error(f"Error executing {func_name}: {str(e)}")
                execution_results[func_name] = False
                # Add error info to the indicators dictionary
                all_indicators[func_name] = {"error": str(e)}
        
        # Log all available indicators for debugging
        logger.info(f"All available indicator keys: {list(all_indicators.keys())}")
        
        # Return the collected indicators
        return all_indicators
    except Exception as e:
        # Handle any unexpected errors in the main function
        logger.error(f"Unexpected error in calculate_technical_indicators_parallel: {str(e)}")
        return {"error": str(e)}

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