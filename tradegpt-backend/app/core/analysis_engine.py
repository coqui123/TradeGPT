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
    ehlers_fisher_transform, chande_momentum_oscillator, elder_triple_screen,
    volume_profile, harmonic_patterns, divergence_scanner,
    stochastic_rsi, elliott_wave_tracker, mean_reversion_index,
    market_breadth_indicators, orderflow_analysis
)
from app.technical_indicators.smart_money_indicators import (
    liquidity_sweep_analysis, order_block_detection, smart_money_analysis,
    cumulative_delta_analysis, volatility_regime_detection, market_depth_analysis,
    funding_liquidation_analysis, cross_asset_correlation
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
    """Calculate technical indicators in parallel for improved performance"""
    logger.info(f"Processing technical analysis with {len(df)} candles")
    
    results = {}
    
    # Create safe accessor functions to handle potential exceptions
    def safe_get_value(series, default=np.nan):
        try:
            return float(series.iloc[-1])
        except (IndexError, ValueError, TypeError, AttributeError):
            return default
    
    # For empty dataframe, return empty results
    if df.empty:
        return {
            "error": "Empty dataframe provided"
        }
    
    # Extract OHLCV data
    try:
        open_prices = df['open']
        high_prices = df['high']
        low_prices = df['low']
        close_prices = df['close']
        volumes = df['volume']
    except KeyError as e:
        return {
            "error": f"Missing required column in dataframe: {str(e)}"
        }
    
    # Define calculation functions for parallel execution
    def calculate_moving_averages():
        """
        Calculate moving averages and update the results dictionary directly.
        This function doesn't return a dictionary; it returns True on success and False on failure.
        The actual results are stored in the results['moving_averages'] dictionary.
        """
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
            results["moving_averages"] = {
                "simple": sma_results,
                "exponential": ema_results
            }
            
            return True
        except Exception as e:
            logger.error(f"Error calculating moving_averages: {str(e)}")
            results["moving_averages"] = {
                "error": str(e),
                "simple": {},
                "exponential": {}
            }
            return False
    
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
        
        # Volume Weighted Average Price (VWAP)
        vwap_val = vwap(df['high'], df['low'], df['close'], df['volume'])
        vwap_results = {
            "VWAP": {
                "value": safe_get_value(vwap_val),
                "data": vwap_val.values.tolist()
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
            **rel_vol_results,
            **vwap_results
        }
    
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
        
        # Create synthetic market breadth data for single instrument analysis
        # These are basic approximations for demonstration purposes
        # In a real market index, you would use actual breadth data from multiple stocks
        price_diff = df['close'].diff()
        advances = pd.Series(np.where(price_diff > 0, 1, 0), index=df.index).rolling(window=5).sum()
        declines = pd.Series(np.where(price_diff < 0, 1, 0), index=df.index).rolling(window=5).sum()
        unchanged = pd.Series(np.where(price_diff == 0, 1, 0), index=df.index).rolling(window=5).sum()
        
        # Calculate market breadth indicators
        mb_indicators = market_breadth_indicators(advances, declines, unchanged)
        
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
                
                # Market Breadth
                "market_breadth": {
                    "advance_decline_line": float(safe_get_value(mb_indicators["advance_decline_line"], 0)),
                    "mcclellan_oscillator": float(safe_get_value(mb_indicators["mcclellan_oscillator"], 0)),
                    "bullish_percent_index": float(safe_get_value(mb_indicators["bullish_percent_index"], 50)),
                    "high_low_index": float(safe_get_value(mb_indicators["high_low_index"], 50))
                },
                
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
                    "hvn_prices": [float(price) for price in vp["hvn"][:3]] if vp["hvn"] else [],  # Include top 3 HVNs
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
            
            # Directional Movement Index
            plus_di, minus_di, adx = directional_movement_index(df['high'], df['low'], df['close'])
            dmi_results = {
                "Plus_DI": {
                    "value": safe_get_value(plus_di),
                    "data": plus_di.values.tolist()
                },
                "Minus_DI": {
                    "value": safe_get_value(minus_di),
                    "data": minus_di.values.tolist()
                },
                "ADX": {
                    "value": safe_get_value(adx),
                    "data": adx.values.tolist()
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
                **dmi_results,
                **elder_ray_results,
                **klinger_results,
                **aroon_results,
                **donchian_results,
                **vortex_results,
                **mass_idx_results,
                **hma_results,
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

    def calculate_smart_money_indicators():
        smart_money_results = {}

        try:
            # Liquidity Sweep Analysis
            sweep_analysis = liquidity_sweep_analysis(
                high_prices, low_prices, close_prices, volumes
            )
            smart_money_results["liquidity_sweeps"] = {
                "high_sweeps_count": len(sweep_analysis['high_sweeps']),
                "low_sweeps_count": len(sweep_analysis['low_sweeps']),
                "current_high_sweep": sweep_analysis['current_high_sweep'],
                "current_low_sweep": sweep_analysis['current_low_sweep'],
                "recent_sweeps": sweep_analysis['high_sweeps'][-3:] + sweep_analysis['low_sweeps'][-3:] 
                                if sweep_analysis['high_sweeps'] or sweep_analysis['low_sweeps'] else []
            }
            
            # Order Block Detection
            order_blocks = order_block_detection(
                open_prices, high_prices, low_prices, close_prices, volumes
            )
            smart_money_results["order_blocks"] = {
                "bullish_blocks_count": len(order_blocks['bullish_order_blocks']),
                "bearish_blocks_count": len(order_blocks['bearish_order_blocks']),
                "active_bullish_count": len(order_blocks['active_bullish_blocks']),
                "active_bearish_count": len(order_blocks['active_bearish_blocks']),
                "recent_bullish": order_blocks['active_bullish_blocks'][-3:] if order_blocks['active_bullish_blocks'] else [],
                "recent_bearish": order_blocks['active_bearish_blocks'][-3:] if order_blocks['active_bearish_blocks'] else []
            }
            
            # Smart Money Concepts
            smc = smart_money_analysis(
                open_prices, high_prices, low_prices, close_prices, volumes
            )
            smart_money_results["fair_value_gaps"] = {
                "bullish_fvg_count": len(smc['bullish_fvg']),
                "bearish_fvg_count": len(smc['bearish_fvg']),
                "equal_highs_count": len(smc['equal_highs']),
                "equal_lows_count": len(smc['equal_lows']),
                "recent_bullish_fvg": smc['bullish_fvg'][-3:] if smc['bullish_fvg'] else [],
                "recent_bearish_fvg": smc['bearish_fvg'][-3:] if smc['bearish_fvg'] else []
            }
            
            # Volatility Regime Detection
            vol_regime = volatility_regime_detection(
                close_prices, high_prices, low_prices
            )
            smart_money_results["volatility_regime"] = vol_regime
            
            # Cumulative Delta Analysis (estimated from price action)
            delta = cumulative_delta_analysis(
                open_prices, high_prices, low_prices, close_prices, volumes
            )
            smart_money_results["cumulative_delta"] = {
                "recent_delta": delta['delta'][-10:] if len(delta['delta']) > 10 else delta['delta'],
                "recent_cumulative_delta": delta['cumulative_delta'][-10:] if len(delta['cumulative_delta']) > 10 else delta['cumulative_delta'],
                "divergence_count": len(delta['delta_divergences']),
                "recent_divergences": delta['delta_divergences'][-3:] if delta['delta_divergences'] else []
            }
            
            # Market Depth Analysis - if order book data is available 
            try:
                # Use estimated bid/ask levels based on recent price action
                current_price = close_prices.iloc[-1]
                range_high = high_prices.iloc[-20:].max()
                range_low = low_prices.iloc[-20:].min()
                
                # Create synthetic levels for demonstration
                bid_levels = [current_price * (1 - 0.001 * i) for i in range(1, 6)]
                ask_levels = [current_price * (1 + 0.001 * i) for i in range(1, 6)]
                level_volumes = [volumes.iloc[-1] * (0.9 ** i) for i in range(10)]
                
                market_depth = market_depth_analysis(
                    float(current_price), bid_levels, ask_levels, level_volumes
                )
                smart_money_results["market_depth"] = {
                    "bid_ask_imbalance": market_depth['bid_ask_imbalance'],
                    "buy_pressure": market_depth['buy_pressure'],
                    "sell_pressure": market_depth['sell_pressure'],
                    "support_zones": [level['price'] for level in market_depth['support_clusters'][:3]] if market_depth['support_clusters'] else [],
                    "resistance_zones": [level['price'] for level in market_depth['resistance_clusters'][:3]] if market_depth['resistance_clusters'] else []
                }
            except Exception as e:
                logger.error(f"Error calculating market depth analysis: {str(e)}")
            
            # Funding and Liquidation Analysis (for crypto)
            try:
                # Create synthetic funding and liquidation data for demonstration
                current_price = close_prices.iloc[-1]
                
                # Simple oscillator based on price momentum
                mom = close_prices.pct_change(5)
                funding_rates = mom.rolling(8).mean() * 0.01  # Scale to typical funding rate ranges
                
                # Create synthetic liquidation data
                long_liqs = pd.Series((close_prices.diff() < 0) * volumes * 0.1, index=close_prices.index)
                short_liqs = pd.Series((close_prices.diff() > 0) * volumes * 0.1, index=close_prices.index)
                
                # Synthetic open interest
                open_interest = volumes.rolling(14).sum()
                
                funding_analysis = funding_liquidation_analysis(
                    float(current_price), funding_rates, long_liqs, short_liqs, open_interest
                )
                smart_money_results["funding_liquidation"] = {
                    "funding_trend": funding_analysis['funding_trend'],
                    "current_funding": funding_analysis['current_funding'],
                    "market_sentiment": funding_analysis['market_sentiment'],
                    "open_interest_change": funding_analysis['open_interest_change'],
                    "long_liquidation_clusters": len(funding_analysis['long_liquidation_clusters']),
                    "short_liquidation_clusters": len(funding_analysis['short_liquidation_clusters'])
                }
            except Exception as e:
                logger.error(f"Error calculating funding and liquidation analysis: {str(e)}")
            
            # Cross Asset Correlation
            try:
                # Create synthetic related assets by adding noise to the main asset
                main_asset_price = close_prices
                related_assets = {
                    'related1': main_asset_price * (1 + np.random.normal(0, 0.01, len(main_asset_price))),
                    'related2': main_asset_price * (1 - np.random.normal(0, 0.01, len(main_asset_price))),
                    'related3': main_asset_price * (1 + np.sin(np.arange(len(main_asset_price)) * 0.1) * 0.05)
                }
                
                correlation_analysis = cross_asset_correlation(
                    main_asset_price, related_assets, [14, 30]
                )
                smart_money_results["cross_asset_correlation"] = {
                    "strongest_positive": correlation_analysis['strongest_positive'],
                    "strongest_negative": correlation_analysis['strongest_negative'],
                    "correlations_14d": {k: v['correlation'] for k, v in correlation_analysis['correlations'].get('14d', {}).items()}
                }
            except Exception as e:
                logger.error(f"Error calculating cross asset correlation: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error calculating smart money indicators: {str(e)}")
            smart_money_results["error"] = str(e)
            
        return smart_money_results
    
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
                    
                    # Add additional context to help the LLM understand the Elliott Wave pattern
                    # This makes it easier for the LLM to interpret the pattern even with incomplete data
                    if elliott_waves['confidence'] > 0:
                        advanced_patterns_result['elliott_wave']['pattern_quality'] = (
                            "high" if elliott_waves['confidence'] > 0.7 else 
                            "medium" if elliott_waves['confidence'] > 0.4 else 
                            "low"
                        )
                        
                        # Add a simplified interpretation
                        wave_pos = elliott_waves['current_position']
                        if "undefined" in wave_pos or "No clear" in wave_pos:
                            advanced_patterns_result['elliott_wave']['interpretation'] = "Elliott Wave pattern is unclear or developing"
                        elif "Impulse wave" in wave_pos:
                            advanced_patterns_result['elliott_wave']['interpretation'] = f"In an impulse move ({wave_pos})"
                        elif "Corrective" in wave_pos:
                            advanced_patterns_result['elliott_wave']['interpretation'] = f"In a corrective move ({wave_pos})"
                        else:
                            advanced_patterns_result['elliott_wave']['interpretation'] = wave_pos
                    else:
                        advanced_patterns_result['elliott_wave']['interpretation'] = "No clear Elliott Wave pattern detected"
                        
                except Exception as e:
                    logger.error(f"Error calculating Elliott Wave: {str(e)}")
                    advanced_patterns_result['elliott_wave'] = {
                        "error": str(e),
                        "interpretation": "Elliott Wave analysis failed due to insufficient or invalid data",
                        "confidence": 0,
                        "wave_count": 0
                    }
            
            # Divergence Scanning between price and RSI
            try:
                # Get RSI values from results dictionary if available, otherwise pass None
                rsi_values = results.get('RSI', {}).get('value') if 'RSI' in results and isinstance(results['RSI'], dict) else None
                
                # If RSI values are available as a series, use them; otherwise, provide a single value
                if rsi_values is not None and not isinstance(rsi_values, (pd.Series, list)):
                    rsi_values = [rsi_values]  # Convert single value to list
                
                divergences = calculate_divergences(df['close'], rsi_values)
                advanced_patterns_result['divergences'] = divergences
            except Exception as e:
                logger.error(f"Error calculating divergences: {str(e)}")
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
            if len(df) >= 30 and 'volume' in df.columns:
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
            
            # Update the main indicator results dictionary with the advanced patterns
            results.update(advanced_patterns_result)
            
            return advanced_patterns_result
        except Exception as e:
            logger.error(f"Error in calculate_advanced_patterns: {str(e)}")
            return {"error": str(e)}

    # Execute calculations in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        calculations = {
            "moving_averages": executor.submit(calculate_moving_averages),
            "momentum": executor.submit(calculate_momentum),
            "volume": executor.submit(calculate_volume_analysis),
            "volatility": executor.submit(calculate_volatility),
            "fibonacci": executor.submit(calculate_fibonacci_levels),
            "market_structure": executor.submit(calculate_market_structure),
            "multi_timeframe": executor.submit(calculate_multi_timeframe_analysis),
            "additional": executor.submit(calculate_additional_indicators),
            "advanced_patterns": executor.submit(calculate_advanced_patterns),
            "smart_money": executor.submit(calculate_smart_money_indicators)
        }
        
        # Collect all results
        for name, future in calculations.items():
            try:
                result = future.result()
                if isinstance(result, dict):
                    results.update(result)
                elif result is True:
                    # The calculation succeeded but didn't return a dictionary
                    # The results were likely already added to the main results dictionary
                    pass
                elif result is False:
                    # The calculation failed, ensure there's an error message
                    if f"{name}_error" not in results:
                        results[f"{name}_error"] = "Calculation returned False without setting an error message"
                else:
                    # Unexpected result type
                    logger.warning(f"Calculation {name} returned unexpected result type: {type(result)}")
            except Exception as e:
                logger.error(f"Error in {name} calculation: {str(e)}")
                results[f"{name}_error"] = str(e)
    
    # Add execution info
    results["_execution_info"] = {
        "candles_processed": len(df),
        "period": f"{df.index[0]} to {df.index[-1]}" if len(df.index) > 0 else "N/A",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    return results

def format_indicators_for_llm(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format technical indicators into a structure optimized for LLM interpretation.
    This creates a layered, contextualized representation that's easier for AI models to understand.
    
    Parameters:
    -----------
    indicators: Dict[str, Any]
        The raw technical indicators calculated by calculate_technical_indicators_parallel
        
    Returns:
    --------
    Dict[str, Any]: Reformatted indicators with additional context and interpretation
    """
    if not indicators or not isinstance(indicators, dict):
        return {"error": "No valid indicators provided"}
    
    # Initialize result structure
    formatted = {
        "overview": {
            "market_bias": "neutral",
            "timeframe_alignment": {},
            "volatility_regime": "normal",
            "institutional_activity": {"detected": False, "description": "No significant institutional activity detected"},
            "key_levels": {"support": [], "resistance": []}
        },
        "trend_indicators": {},
        "momentum_indicators": {},
        "volatility_indicators": {},
        "volume_indicators": {},
        "support_resistance": {},
        "smart_money_concepts": {},
        "pattern_recognition": {},
        "divergences": []
    }
    
    # Helper function to recursively filter out None values from dictionaries and lists
    def filter_none_values(obj):
        if isinstance(obj, dict):
            return {k: filter_none_values(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [filter_none_values(item) for item in obj if item is not None]
        else:
            return obj
    
    # Extract market structure info if available
    market_structure = indicators.get("Market_Structure", {})
    if isinstance(market_structure, dict):
        # Set market bias
        trend = market_structure.get("trend_direction", "").lower()
        if "bullish" in trend:
            formatted["overview"]["market_bias"] = "bullish"
        elif "bearish" in trend:
            formatted["overview"]["market_bias"] = "bearish"
        
        # Add key support/resistance levels
        if "pivot" in market_structure:
            formatted["overview"]["key_levels"]["support"].extend([
                market_structure.get("s1", None),
                market_structure.get("s2", None),
                market_structure.get("s3", None)
            ])
            formatted["overview"]["key_levels"]["resistance"].extend([
                market_structure.get("r1", None),
                market_structure.get("r2", None),
                market_structure.get("r3", None)
            ])
            # Filter out None values
            formatted["overview"]["key_levels"]["support"] = [s for s in formatted["overview"]["key_levels"]["support"] if s is not None]
            formatted["overview"]["key_levels"]["resistance"] = [r for r in formatted["overview"]["key_levels"]["resistance"] if r is not None]
        
        # Add market breadth if available
        if "market_breadth" in market_structure:
            formatted["overview"]["market_internals"] = market_structure["market_breadth"]
    
    # Extract smart money indicators if available
    if "liquidity_sweeps" in indicators:
        formatted["smart_money_concepts"]["liquidity_sweeps"] = filter_none_values(indicators["liquidity_sweeps"])
        if indicators["liquidity_sweeps"].get("current_high_sweep") or indicators["liquidity_sweeps"].get("current_low_sweep"):
            formatted["overview"]["institutional_activity"]["detected"] = True
            formatted["overview"]["institutional_activity"]["description"] = "Recent liquidity sweep detected"
    
    if "order_blocks" in indicators:
        formatted["smart_money_concepts"]["order_blocks"] = filter_none_values(indicators["order_blocks"])
        if indicators["order_blocks"].get("active_bullish_count", 0) > 0 or indicators["order_blocks"].get("active_bearish_count", 0) > 0:
            formatted["overview"]["institutional_activity"]["detected"] = True
            formatted["overview"]["institutional_activity"]["description"] = "Active order blocks detected"
    
    if "fair_value_gaps" in indicators:
        formatted["smart_money_concepts"]["fair_value_gaps"] = filter_none_values(indicators["fair_value_gaps"])
    
    # Extract volatility regime if available
    if "volatility_regime" in indicators:
        regime = indicators["volatility_regime"].get("regime", "normal_volatility")
        formatted["overview"]["volatility_regime"] = regime.replace("_volatility", "")
    
    # Extract moving averages
    if "moving_averages" in indicators:
        moving_averages = indicators["moving_averages"]
        ma_trends = {}
        
        # Process simple moving averages
        if "simple" in moving_averages:
            simple_mas = moving_averages["simple"]
            for period in ["20", "50", "100", "200"]:
                if period in simple_mas:
                    ma_name = f"SMA_{period}"
                    value = simple_mas[period].get("value")
                    if value is not None:  # Only add non-None values
                        ma_trends[ma_name] = value
        
        # Process exponential moving averages  
        if "exponential" in moving_averages:
            exp_mas = moving_averages["exponential"]
            for period in ["20", "50", "100", "200"]:
                if period in exp_mas:
                    ma_name = f"EMA_{period}"
                    value = exp_mas[period].get("value")
                    if value is not None:  # Only add non-None values
                        ma_trends[ma_name] = value
        
        # Add moving average trends
        if ma_trends:  # Only add if there are non-empty values
            formatted["trend_indicators"]["moving_averages"] = ma_trends
    
    # Process RSI
    if "RSI" in indicators:
        rsi_data = indicators["RSI"]
        rsi_value = None
        
        if isinstance(rsi_data, dict) and "value" in rsi_data:
            rsi_value = rsi_data["value"]
        elif isinstance(rsi_data, (int, float)):
            rsi_value = rsi_data
        
        if rsi_value is not None:
            formatted["momentum_indicators"]["rsi"] = {
                "value": rsi_value,
                "state": "overbought" if rsi_value > 70 else "oversold" if rsi_value < 30 else "neutral"
            }
    
    # Process MACD
    if "MACD" in indicators:
        macd_data = indicators["MACD"]
        if isinstance(macd_data, dict):
            macd_line = macd_data.get("line")
            signal_line = macd_data.get("signal")
            histogram = macd_data.get("histogram")
            
            if all(x is not None for x in [macd_line, signal_line, histogram]):
                formatted["momentum_indicators"]["macd"] = {
                    "line": macd_line,
                    "signal": signal_line,
                    "histogram": histogram,
                    "signal_direction": "bullish" if histogram > 0 else "bearish",
                    "crossover": "bullish" if macd_line > signal_line else "bearish"
                }
    
    # Process Bollinger Bands
    if "Bollinger_Bands" in indicators:
        bb_data = indicators["Bollinger_Bands"]
        if isinstance(bb_data, dict):
            upper = bb_data.get("upper")
            middle = bb_data.get("middle")
            lower = bb_data.get("lower")
            
            if all(x is not None for x in [upper, middle, lower]):
                price = indicators.get("close", [])[-1] if indicators.get("close") else None
                formatted["volatility_indicators"]["bollinger_bands"] = {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "width": (upper - lower) / middle if middle != 0 else 0,
                }
                
                # Only add price_position if we have the price
                if price is not None:
                    formatted["volatility_indicators"]["bollinger_bands"]["price_position"] = (
                        "upper_band" if price >= upper else 
                        "lower_band" if price <= lower else "middle_range"
                    )
    
    # Process volume indicators
    volume_indicators = {}
    for indicator in ["OBV", "CMF", "MFI", "Chaikin_Oscillator", "VWAP"]:
        if indicator in indicators and isinstance(indicators[indicator], dict) and "value" in indicators[indicator]:
            value = indicators[indicator]["value"]
            if value is not None:  # Only add non-None values
                volume_indicators[indicator] = value
    
    if volume_indicators:
        formatted["volume_indicators"] = volume_indicators
    
    # Process Smart Money indicators
    if "funding_liquidation" in indicators:
        funding_data = indicators["funding_liquidation"]
        if isinstance(funding_data, dict):
            formatted["smart_money_concepts"]["funding_sentiment"] = {
                "trend": funding_data.get("funding_trend", "neutral"),
                "market_sentiment": funding_data.get("market_sentiment", "neutral"),
                "open_interest_change": funding_data.get("open_interest_change", 0)
            }
    
    # Process cross-asset correlation if available
    if "cross_asset_correlation" in indicators:
        correlation_data = indicators["cross_asset_correlation"]
        if isinstance(correlation_data, dict):
            formatted["smart_money_concepts"]["correlations"] = {
                "strongest_positive": correlation_data.get("strongest_positive", {}),
                "strongest_negative": correlation_data.get("strongest_negative", {})
            }
    
    # Process divergences (specific price/indicator divergences)
    if "cumulative_delta" in indicators and "delta_divergences" in indicators["cumulative_delta"]:
        for div in indicators["cumulative_delta"]["delta_divergences"]:
            if div and isinstance(div, dict):
                formatted["divergences"].append({
                    "type": "delta_price",
                    "bias": div.get("type", ""),
                    "price": div.get("price"),
                    "description": f"{div.get('type', '').capitalize()} divergence between price and volume delta"
                })
    
    # Add divergence scanner results if available
    if "divergences" in indicators:
        divs = indicators["divergences"]
        if isinstance(divs, dict):
            for div_type, div_list in divs.items():
                if div_list and isinstance(div_list, list):
                    for div in div_list:
                        if div and isinstance(div, dict):
                            formatted["divergences"].append({
                                "type": div_type,
                                "indicator": div.get("indicator", ""),
                                "bias": div.get("type", ""),
                                "strength": div.get("strength", 0),
                                "description": f"{div.get('type', '').capitalize()} divergence between price and {div.get('indicator', '')}"
                            })
    
    # Multi-timeframe alignment if available
    if "multi_timeframe" in indicators:
        mtf = indicators["multi_timeframe"]
        if isinstance(mtf, dict):
            for tf, data in mtf.items():
                if isinstance(data, dict) and "trend" in data:
                    formatted["overview"]["timeframe_alignment"][tf] = data["trend"]
    
    # Process pattern recognition results
    patterns = []
    
    # Check for harmonic patterns
    if "Harmonic_Patterns" in indicators:
        harmonics = indicators["Harmonic_Patterns"]
        if isinstance(harmonics, dict):
            for pattern_type, pattern_data in harmonics.items():
                if pattern_data and isinstance(pattern_data, (list, dict)):
                    if isinstance(pattern_data, list):
                        for p in pattern_data:
                            if p and isinstance(p, dict) and "type" in p and "completion" in p:
                                patterns.append({
                                    "type": "harmonic",
                                    "pattern": p["type"],
                                    "completion": p["completion"],
                                    "bias": "bullish" if "bullish" in p["type"].lower() else "bearish" if "bearish" in p["type"].lower() else "neutral"
                                })
                    elif "type" in pattern_data and "completion" in pattern_data:
                        patterns.append({
                            "type": "harmonic",
                            "pattern": pattern_data["type"],
                            "completion": pattern_data["completion"],
                            "bias": "bullish" if "bullish" in pattern_data["type"].lower() else "bearish" if "bearish" in pattern_data["type"].lower() else "neutral"
                        })
    
    # Process Elliott Wave patterns
    if "elliott_wave" in indicators:
        elliott_data = indicators["elliott_wave"]
        if isinstance(elliott_data, dict):
            # Create base pattern data
            ew_pattern = {
                "type": "elliott_wave",
                "pattern": "Elliott Wave",
                "confidence": elliott_data.get("confidence", 0),
                "position": elliott_data.get("current_position", "undefined"),
                "bias": "undefined"  # Default value
            }
            
            # Add interpretation if available
            if "interpretation" in elliott_data:
                ew_pattern["interpretation"] = elliott_data["interpretation"]
                ew_pattern["quality"] = elliott_data.get("pattern_quality", "low")
                
                # Try to determine trading bias from the current position
                position = elliott_data.get("current_position", "").lower()
                if position:
                    if "impulse" in position and "5" in position:
                        ew_pattern["bias"] = "reversal expected"
                    elif "impulse" in position and any(str(i) in position for i in [1, 2, 3, 4]):
                        ew_pattern["bias"] = "trend continuation likely"
                    elif "corrective" in position and "complete" in position:
                        ew_pattern["bias"] = "trend resumption likely"
                
                # Add wave structures if available
                impulse_waves = elliott_data.get("impulse_waves", [])
                if impulse_waves and isinstance(impulse_waves, list):
                    valid_waves = [w for w in impulse_waves if w is not None and isinstance(w, dict)]
                    if valid_waves:
                        ew_pattern["impulse_structure"] = valid_waves
                
                corrective_waves = elliott_data.get("corrective_waves", [])
                if corrective_waves and isinstance(corrective_waves, list):
                    valid_waves = [w for w in corrective_waves if w is not None and isinstance(w, dict)]
                    if valid_waves:
                        ew_pattern["corrective_structure"] = valid_waves
            else:
                # Use confidence to create interpretation if not explicitly provided
                confidence = elliott_data.get("confidence", 0)
                if confidence > 0.7:
                    ew_pattern["interpretation"] = "Clear Elliott Wave pattern"
                elif confidence > 0.3:
                    ew_pattern["interpretation"] = "Developing Elliott Wave pattern"
                else:
                    ew_pattern["interpretation"] = "Unclear Elliott Wave pattern"
            
            # Add the pattern to the list
            patterns.append(ew_pattern)
    
    # Check for candlestick patterns
    if "candlestick_patterns" in indicators:
        candle_patterns = indicators["candlestick_patterns"]
        if isinstance(candle_patterns, dict):
            for pattern, value in candle_patterns.items():
                if value != 0:  # 0 means no pattern
                    patterns.append({
                        "type": "candlestick",
                        "pattern": pattern,
                        "bias": "bullish" if value > 0 else "bearish"
                    })
    
    # Add patterns to formatted output
    if patterns:
        formatted["pattern_recognition"]["patterns"] = patterns
    
    # Final cleanup - recursively remove empty dicts, lists, and None values
    formatted = filter_none_values(formatted)
    
    # Remove empty dictionaries and lists to save tokens
    for section in list(formatted.keys()):
        if isinstance(formatted[section], dict) and not formatted[section]:
            del formatted[section]
        elif isinstance(formatted[section], list) and not formatted[section]:
            del formatted[section]
    
    return formatted

def add_smart_money_context(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches indicator data with contextual explanations to help the LLM better understand
    and interpret institutional activity in the market.
    
    Parameters:
    -----------
    indicators: Dict[str, Any]
        The indicators dictionary to enrich with context
        
    Returns:
    --------
    Dict[str, Any]: The enriched indicators with added context
    """
    if not indicators or not isinstance(indicators, dict):
        return indicators
    
    # Deep copy to avoid modifying the original
    enriched = indicators.copy()
    
    # Add context section
    enriched["context"] = {
        "methodology": {},
        "indicator_explanations": {},
        "actionable_insights": []
    }
    
    # Add smart money methodologies
    methodologies = {
        "liquidity_sweep": "A liquidity sweep occurs when price briefly breaks a significant level (high/low) to trigger stop losses or limit orders, then quickly reverses. This is a common institutional tactic to accumulate or distribute at favorable prices.",
        
        "order_block": "Order blocks are zones where large institutional orders were placed that led to a significant move. Bullish order blocks form at the base of upward moves, while bearish order blocks form at the top of downward moves. These areas often act as strong support/resistance when retested.",
        
        "fair_value_gap": "Fair Value Gaps (FVGs) are significant imbalances in price where a sharp movement creates a gap in fair value. These areas often get 'filled' as price returns to establish fair value in the skipped region.",
        
        "delta_divergence": "Delta divergence occurs when the directional flow of money (cumulative delta) contradicts price movement. For example, if price makes a higher high but cumulative delta makes a lower high, this suggests bearish divergence."
    }
    
    enriched["context"]["methodology"] = methodologies
    
    # Add contextual explanations for available indicators
    explanations = {}
    action_insights = []
    
    # Liquidity sweeps explanations
    if "liquidity_sweeps" in indicators:
        sweeps = indicators["liquidity_sweeps"]
        if sweeps:
            explanations["liquidity_sweeps"] = "Identifies areas where price has briefly broken significant levels, triggering stop losses, before reversing. Current sweeps indicate recent tests of liquidity that could lead to reversals."
            
            # Add actionable insights
            if sweeps.get("current_high_sweep", False):
                action_insights.append("A high sweep is currently in progress - watch for a potential bearish reversal if price fails to maintain above the swept level.")
            
            if sweeps.get("current_low_sweep", False):
                action_insights.append("A low sweep is currently in progress - watch for a potential bullish reversal if price fails to maintain below the swept level.")
            
            if sweeps.get("high_sweeps_count", 0) > 0 or sweeps.get("low_sweeps_count", 0) > 0:
                action_insights.append(f"Multiple liquidity sweeps detected ({sweeps.get('high_sweeps_count', 0)} high, {sweeps.get('low_sweeps_count', 0)} low) - indicates active institutional manipulation in this market.")
    
    # Order blocks explanations
    if "order_blocks" in indicators:
        blocks = indicators["order_blocks"]
        if blocks:
            explanations["order_blocks"] = "Zones where significant institutional buying or selling occurred that led to strong directional moves. Active blocks are those that haven't been broken and may provide support/resistance."
            
            # Add actionable insights
            if blocks.get("active_bullish_count", 0) > 0:
                action_insights.append(f"Active bullish order blocks ({blocks.get('active_bullish_count', 0)}) indicate potential support zones where institutional buying previously occurred.")
            
            if blocks.get("active_bearish_count", 0) > 0:
                action_insights.append(f"Active bearish order blocks ({blocks.get('active_bearish_count', 0)}) indicate potential resistance zones where institutional selling previously occurred.")
    
    # Fair value gaps explanations
    if "fair_value_gaps" in indicators:
        fvg = indicators["fair_value_gaps"]
        if fvg:
            explanations["fair_value_gaps"] = "Areas where price moved so quickly it created imbalances or 'gaps' in fair value. These gaps tend to eventually get 'filled' as price returns to establish fair value."
            
            # Add actionable insights
            bullish_count = fvg.get("bullish_fvg_count", 0)
            bearish_count = fvg.get("bearish_fvg_count", 0)
            
            if bullish_count > 0:
                action_insights.append(f"Bullish fair value gaps ({bullish_count}) indicate potential support zones below current price that may attract price action to 'fill the gap'.")
            
            if bearish_count > 0:
                action_insights.append(f"Bearish fair value gaps ({bearish_count}) indicate potential resistance zones above current price that may attract price action to 'fill the gap'.")
    
    # Cumulative delta explanations
    if "cumulative_delta" in indicators:
        delta = indicators["cumulative_delta"]
        if delta:
            explanations["cumulative_delta"] = "Tracks the net buying vs. selling pressure over time. Divergences between price and delta can reveal hidden strength or weakness not obvious from price action alone."
            
            # Add actionable insights
            divergence_count = len(delta.get("delta_divergences", []))
            if divergence_count > 0:
                recent_divs = delta.get("delta_divergences", [])[-3:] if delta.get("delta_divergences") else []
                div_types = [d.get("type", "") for d in recent_divs if d and isinstance(d, dict)]
                
                bullish_count = sum(1 for t in div_types if t == "bullish")
                bearish_count = sum(1 for t in div_types if t == "bearish")
                
                if bullish_count > bearish_count:
                    action_insights.append(f"Recent bullish delta divergences detected - price may be forming a bottom despite downward price movement.")
                elif bearish_count > bullish_count:
                    action_insights.append(f"Recent bearish delta divergences detected - price may be forming a top despite upward price movement.")
    
    # Market depth explanations
    if "market_depth" in indicators:
        depth = indicators["market_depth"]
        if depth:
            explanations["market_depth"] = "Analyzes the order book to identify imbalances in buying vs. selling pressure and clusters of orders that may act as support/resistance."
            
            # Add actionable insights
            if depth.get("bid_ask_imbalance", 1.0) > 1.5:
                action_insights.append(f"Strong bid pressure with bid/ask imbalance of {depth.get('bid_ask_imbalance', 1.0):.2f} - indicating potential buying pressure exceeding selling pressure.")
            elif depth.get("bid_ask_imbalance", 1.0) < 0.67:
                action_insights.append(f"Strong ask pressure with bid/ask imbalance of {depth.get('bid_ask_imbalance', 1.0):.2f} - indicating potential selling pressure exceeding buying pressure.")
    
    # Funding and liquidation analysis explanations
    if "funding_liquidation" in indicators:
        funding = indicators["funding_liquidation"]
        if funding:
            explanations["funding_liquidation"] = "For cryptocurrency futures markets, analyzes funding rates and liquidation levels to understand market sentiment and potential liquidation cascades."
            
            # Add actionable insights
            market_sentiment = funding.get("market_sentiment", "neutral")
            oi_change = funding.get("open_interest_change", 0)
            
            if market_sentiment != "neutral":
                action_insights.append(f"Funding rates indicate {market_sentiment} market sentiment with {oi_change:.1f}% open interest change - suggesting directional bias from futures markets.")
            
            long_liq = funding.get("long_liquidation_clusters", 0)
            short_liq = funding.get("short_liquidation_clusters", 0)
            
            if long_liq > 0 or short_liq > 0:
                action_insights.append(f"Liquidation clusters detected ({long_liq} long, {short_liq} short) - indicating potential stop hunt targets for institutions.")
    
    # Add the explanations and insights to the enriched indicators
    enriched["context"]["indicator_explanations"] = explanations
    enriched["context"]["actionable_insights"] = action_insights
    
    return enriched

def optimize_indicators_for_llm(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimizes technical indicators for LLM processing by removing null values and
    unnecessary data to reduce token usage.
    
    Parameters:
    -----------
    indicators: Dict[str, Any]
        The technical indicators dictionary
        
    Returns:
    --------
    Dict[str, Any]: Optimized indicators with null values removed
    """
    if not indicators or not isinstance(indicators, dict):
        return {}
    
    # Helper function to recursively filter null values
    def filter_null_values(obj):
        if obj is None:
            return None
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                filtered_v = filter_null_values(v)
                if filtered_v is not None:
                    result[k] = filtered_v
            return result if result else None
        elif isinstance(obj, list):
            filtered_list = [filter_null_values(item) for item in obj if item is not None]
            filtered_list = [item for item in filtered_list if item is not None]
            return filtered_list if filtered_list else None
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj
    
    # Apply filtering to the entire indicators dictionary
    optimized = filter_null_values(indicators)
    
    # Remove historical data arrays that consume a lot of tokens
    # but aren't as useful for LLM analysis
    if optimized and isinstance(optimized, dict):
        # Keep only the most recent values for indicator arrays
        for key, value in optimized.items():
            if isinstance(value, dict):
                # Special handling for Elliott Wave data
                if key == "elliott_wave":
                    # Ensure we preserve important information while trimming excess
                    if "impulse_waves" in value and isinstance(value["impulse_waves"], list) and len(value["impulse_waves"]) > 0:
                        # Keep only the essential data from impulse waves
                        simplified_waves = []
                        for wave in value["impulse_waves"]:
                            if wave and isinstance(wave, dict):
                                simplified_wave = {
                                    "wave": wave.get("wave"),
                                    "price": wave.get("price")
                                }
                                simplified_waves.append(simplified_wave)
                        value["impulse_waves"] = simplified_waves
                    
                    if "corrective_waves" in value and isinstance(value["corrective_waves"], list) and len(value["corrective_waves"]) > 0:
                        # Keep only the essential data from corrective waves
                        simplified_waves = []
                        for wave in value["corrective_waves"]:
                            if wave and isinstance(wave, dict):
                                simplified_wave = {
                                    "wave": wave.get("wave"),
                                    "price": wave.get("price")
                                }
                                simplified_waves.append(simplified_wave)
                        value["corrective_waves"] = simplified_waves
                        
                    # Add an interpretation field if it doesn't exist
                    if "interpretation" not in value:
                        confidence = value.get("confidence", 0)
                        current_position = value.get("current_position", "undefined")
                        if confidence > 0.7:
                            value["interpretation"] = "Clear Elliott Wave pattern"
                        elif confidence > 0.3:
                            value["interpretation"] = "Developing Elliott Wave pattern"
                        else:
                            value["interpretation"] = "Unclear Elliott Wave pattern"
                        
                        # Add pattern quality
                        value["pattern_quality"] = (
                            "high" if confidence > 0.7 else 
                            "medium" if confidence > 0.4 else 
                            "low"
                        )
                else:
                    # For other nested dictionaries, check for data arrays
                    for k, v in list(value.items()):
                        # If a key contains 'data' and has a list longer than 10 items, truncate it
                        if k == 'data' and isinstance(v, list) and len(v) > 10:
                            value[k] = v[-10:]  # Keep only the most recent 10 values
                        # For nested values that are dictionaries, check recursively
                        elif isinstance(v, dict):
                            for subk, subv in list(v.items()):
                                if subk == 'data' and isinstance(subv, list) and len(subv) > 10:
                                    v[subk] = subv[-10:]
    
    return optimized

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
    
    # Optimize indicators for LLM
    optimized_indicators = optimize_indicators_for_llm(indicators)
    
    # Prepare technical indicators summary
    indicators_summary = ""
    if optimized_indicators:
        # Moving Averages
        ma_summary = "Moving Averages:\n"
        for ma_type in ["SMA_20", "SMA_50", "SMA_200", "EMA_20"]:
            if ma_type in optimized_indicators:
                ma_val = optimized_indicators[ma_type]["value"]
                ma_summary += f"- {ma_type}: {ma_val}\n"
        
        # RSI
        rsi_indicator = optimized_indicators.get("RSI", {})
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
        macd_info = optimized_indicators.get("MACD", {})
        
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
        bb_info = optimized_indicators.get("Bollinger_Bands", {})
        
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
        market_structure = optimized_indicators.get("Market_Structure", {})
        
        # Handle different possible formats for Market Structure
        if isinstance(market_structure, dict):
            trend_direction = market_structure.get("trend_direction", "N/A")
        else:
            # If Market_Structure is not a dictionary, set default trend direction
            trend_direction = "N/A"
            # Log for debugging
            logging.warning(f"Market_Structure is not in expected format: {type(market_structure)}")
        
        # Average True Range (ATR)
        atr_indicator = optimized_indicators.get("ATR", {})
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

def calculate_divergences(price, rsi_values):
    """
    Calculate price/indicator divergences
    
    Parameters:
    -----------
    price : pd.Series or list
        Price data
    rsi_values : pd.Series or list
        RSI indicator values
        
    Returns:
    --------
    dict: Dictionary containing detected divergences
    """
    try:
        import pandas as pd
        import numpy as np
        from app.technical_indicators.advanced_indicators import divergence_scanner
        
        # Convert to pandas Series if they aren't already
        if not isinstance(price, pd.Series):
            price = pd.Series(price)
        
        # If RSI values aren't provided, calculate them
        if rsi_values is None:
            from app.technical_indicators.basic_indicators import relative_strength_index
            rsi_values = relative_strength_index(price)
        elif not isinstance(rsi_values, pd.Series):
            rsi_values = pd.Series(rsi_values)
        
        # Scan for divergences using the specialized function
        divergences = divergence_scanner(price, rsi_values)
        
        # Return formatted divergences
        return {
            'regular_bullish': divergences.get('regular_bullish', []),
            'regular_bearish': divergences.get('regular_bearish', []),
            'hidden_bullish': divergences.get('hidden_bullish', []),
            'hidden_bearish': divergences.get('hidden_bearish', []),
            'strength': divergences.get('strength', 0)
        }
    except Exception as e:
        logger.error(f"Error in calculate_divergences: {str(e)}")
        return {
            'regular_bullish': [],
            'regular_bearish': [],
            'hidden_bullish': [],
            'hidden_bearish': [],
            'strength': 0,
            'error': str(e)
        }