"""
Test script to verify the implementation of technical indicators
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Create a log file
log_file = open("indicator_test_log.txt", "w")

def log(message):
    """Write message to log file and print to console"""
    log_file.write(message + "\n")
    print(message)

# Ensure the app directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

log("=" * 50)
log("TECHNICAL INDICATORS TEST SUITE")
log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 50)

# Create sample data
try:
    # Create sample data with trending and ranging periods
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2022-01-01', periods=200)
    
    # Create a trending series followed by a ranging series
    trend = np.linspace(0, 15, 100)  # Uptrend
    range_series = np.random.randn(100) * 2  # Ranging with some volatility
    
    # Combine for price series with some randomness
    combined_series = np.concatenate([trend, range_series])
    noise = np.random.randn(200) * 0.5
    
    # Create OHLCV data
    data = {
        'open': combined_series + 100 + noise,
        'high': combined_series + 102 + np.abs(noise) * 1.5,
        'low': combined_series + 98 - np.abs(noise) * 1.5,
        'close': combined_series + 100 + noise,
        'volume': np.random.randint(1000, 10000, 200) * (1 + np.abs(noise)/10)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Create weekly data for multi-timeframe testing
    weekly_df = df.resample('W').agg({
        'open': 'first', 
        'high': 'max', 
        'low': 'min', 
        'close': 'last',
        'volume': 'sum'
    })
    
    log("\nTest data created successfully!")
    log(f"Daily data shape: {df.shape}")
    log(f"Weekly data shape: {weekly_df.shape}")
    
except Exception as e:
    log(f"Error creating test data: {str(e)}")
    sys.exit(1)

# Test Basic Indicators
log("\n" + "=" * 50)
log("TESTING BASIC INDICATORS")
log("=" * 50)

try:
    from app.technical_indicators.basic_indicators import (
        macd, bollinger_bands, relative_strength_index, average_true_range,
        pivot_points, elder_ray_index, stochastic_oscillator, on_balance_volume,
        keltner_channel, directional_movement_index, williams_percent_r,
        volume_zone_oscillator, relative_volume, chandelier_exit, market_regime
    )
    log("Successfully imported basic indicators!")
except Exception as e:
    log(f"Error importing basic indicators: {str(e)}")
    sys.exit(1)

# Test trend indicators
try:
    log("\n--- Testing Trend Indicators ---")
    
    # MACD
    macd_line, signal_line, histogram = macd(df['close'])
    log(f"MACD: {macd_line.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}, Histogram: {histogram.iloc[-1]:.4f}")
    
    # Bollinger Bands
    upper_bb, middle_bb, lower_bb = bollinger_bands(df['close'])
    log(f"Bollinger Bands - Upper: {upper_bb.iloc[-1]:.4f}, Middle: {middle_bb.iloc[-1]:.4f}, Lower: {lower_bb.iloc[-1]:.4f}")
    
    # Directional Movement Index
    plus_di, minus_di, adx = directional_movement_index(df['high'], df['low'], df['close'])
    log(f"DMI - +DI: {plus_di.iloc[-1]:.4f}, -DI: {minus_di.iloc[-1]:.4f}, ADX: {adx.iloc[-1]:.4f}")
    
    # Keltner Channel
    upper_kc, middle_kc, lower_kc = keltner_channel(df['high'], df['low'], df['close'])
    log(f"Keltner Channel - Upper: {upper_kc.iloc[-1]:.4f}, Middle: {middle_kc.iloc[-1]:.4f}, Lower: {lower_kc.iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing trend indicators: {str(e)}")

# Test momentum indicators
try:
    log("\n--- Testing Momentum Indicators ---")
    
    # RSI
    rsi = relative_strength_index(df['close'])
    log(f"RSI: {rsi.iloc[-1]:.4f}")
    
    # Stochastic Oscillator
    k, d = stochastic_oscillator(df['high'], df['low'], df['close'])
    log(f"Stochastic Oscillator - %K: {k.iloc[-1]:.4f}, %D: {d.iloc[-1]:.4f}")
    
    # Williams %R
    wr = williams_percent_r(df['high'], df['low'], df['close'])
    log(f"Williams %R: {wr.iloc[-1]:.4f}")
    
    # Elder Ray Index
    bull_power, bear_power = elder_ray_index(df['high'], df['low'], df['close'])
    log(f"Elder Ray - Bull Power: {bull_power.iloc[-1]:.4f}, Bear Power: {bear_power.iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing momentum indicators: {str(e)}")

# Test volume indicators
try:
    log("\n--- Testing Volume Indicators ---")
    
    # On-Balance Volume
    obv = on_balance_volume(df['close'], df['volume'])
    log(f"OBV: {obv.iloc[-1]:.2f}")
    
    # Volume Zone Oscillator
    vzo, vzo_signal = volume_zone_oscillator(df['close'], df['volume'])
    log(f"VZO: {vzo.iloc[-1]:.4f}, Signal: {vzo_signal.iloc[-1]:.4f}")
    
    # Relative Volume
    rel_vol = relative_volume(df['volume'])
    log(f"Relative Volume: {rel_vol.iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing volume indicators: {str(e)}")

# Test volatility and support/resistance indicators
try:
    log("\n--- Testing Volatility & Support/Resistance Indicators ---")
    
    # Average True Range
    atr = average_true_range(df['high'], df['low'], df['close'])
    log(f"ATR: {atr.iloc[-1]:.4f}")
    
    # Pivot Points
    pivot, r1, s1, r2, s2, r3, s3 = pivot_points(df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
    log(f"Pivot Points - Pivot: {pivot:.4f}, R1: {r1:.4f}, S1: {s1:.4f}")
    
    # Chandelier Exit
    long_exit, short_exit = chandelier_exit(df['high'], df['low'], df['close'])
    log(f"Chandelier Exit - Long: {long_exit.iloc[-1]:.4f}, Short: {short_exit.iloc[-1]:.4f}")
    
    # Market Regime
    regime, volatility = market_regime(df['close'])
    log(f"Market Regime: {regime.iloc[-1]:.4f}, Volatility Flag: {volatility.iloc[-1]}")
    
except Exception as e:
    log(f"Error testing volatility indicators: {str(e)}")

# Test Advanced Indicators
log("\n" + "=" * 50)
log("TESTING ADVANCED INDICATORS")
log("=" * 50)

try:
    from app.technical_indicators.advanced_indicators import (
        accumulation_distribution_line, chaikin_oscillator, aroon_indicator, 
        chaikin_money_flow, parabolic_sar, money_flow_index, percentage_price_oscillator,
        donchian_channels, rate_of_change, commodity_channel_index, awesome_oscillator,
        vortex_indicator, true_strength_index, mass_index, hull_moving_average,
        coppock_curve, vwap, klinger_oscillator, ichimoku_cloud, supertrend,
        heikin_ashi, squeeze_momentum, ehlers_fisher_transform, chande_momentum_oscillator,
        camarilla_pivot_points, woodie_pivot_points, demark_pivot_points, elder_triple_screen,
        volume_profile, harmonic_patterns, divergence_scanner,
        # Add new indicators to import
        stochastic_rsi, elliott_wave_tracker, mean_reversion_index, 
        market_breadth_indicators, orderflow_analysis
    )
    log("Successfully imported advanced indicators!")
except Exception as e:
    log(f"Error importing advanced indicators: {str(e)}")

# Test advanced trend indicators
try:
    log("\n--- Testing Advanced Trend Indicators ---")
    
    # Ichimoku Cloud
    try:
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_cloud(df['high'], df['low'], df['close'])
        log(f"Ichimoku - Tenkan: {tenkan.iloc[-1]:.4f}, Kijun: {kijun.iloc[-1]:.4f}")
        log(f"Ichimoku - Senkou A: {senkou_a.iloc[-30]:.4f}, Senkou B: {senkou_b.iloc[-30]:.4f}")
    except Exception as e:
        log(f"Error in Ichimoku Cloud: {str(e)}")
    
    # Supertrend
    try:
        super_trend, super_direction = supertrend(df['high'], df['low'], df['close'])
        log(f"Supertrend - Value: {super_trend.iloc[-1]:.4f}, Direction: {super_direction.iloc[-1]}")
    except Exception as e:
        log(f"Error in Supertrend: {str(e)}")
    
    # Parabolic SAR
    try:
        sar, sar_direction = parabolic_sar(df['high'], df['low'])
        log(f"Parabolic SAR - Value: {sar.iloc[-1]:.4f}, Direction: {sar_direction.iloc[-1]}")
    except Exception as e:
        log(f"Error in Parabolic SAR: {str(e)}")
    
    # Donchian Channels
    try:
        upper_dc, middle_dc, lower_dc = donchian_channels(df['high'], df['low'])
        log(f"Donchian Channels - Upper: {upper_dc.iloc[-1]:.4f}, Middle: {middle_dc.iloc[-1]:.4f}, Lower: {lower_dc.iloc[-1]:.4f}")
    except Exception as e:
        log(f"Error in Donchian Channels: {str(e)}")
    
    # Hull Moving Average
    try:
        hma = hull_moving_average(df['close'])
        log(f"Hull Moving Average: {hma.iloc[-1]:.4f}")
    except Exception as e:
        log(f"Error in Hull Moving Average: {str(e)}")
    
    # Vortex Indicator
    try:
        plus_vi, minus_vi = vortex_indicator(df['high'], df['low'], df['close'])
        log(f"Vortex - +VI: {plus_vi.iloc[-1]:.4f}, -VI: {minus_vi.iloc[-1]:.4f}")
    except Exception as e:
        log(f"Error in Vortex Indicator: {str(e)}")
    
    # Heikin-Ashi
    try:
        ha_open, ha_high, ha_low, ha_close = heikin_ashi(df['open'], df['high'], df['low'], df['close'])
        log(f"Heikin-Ashi - Close: {ha_close.iloc[-1]:.4f}")
    except Exception as e:
        log(f"Error in Heikin-Ashi: {str(e)}")
    
except Exception as e:
    log(f"Error testing advanced trend indicators: {str(e)}")

# Test advanced momentum indicators
try:
    log("\n--- Testing Advanced Momentum Indicators ---")
    
    # Commodity Channel Index
    cci = commodity_channel_index(df['high'], df['low'], df['close'])
    log(f"CCI: {cci.iloc[-1]:.4f}")
    
    # Awesome Oscillator
    ao = awesome_oscillator(df['high'], df['low'])
    log(f"Awesome Oscillator: {ao.iloc[-1]:.4f}")
    
    # Rate of Change
    roc = rate_of_change(df['close'])
    log(f"Rate of Change: {roc.iloc[-1]:.4f}")
    
    # True Strength Index
    tsi = true_strength_index(df['close'])
    log(f"True Strength Index: {tsi.iloc[-1]:.4f}")
    
    # Percentage Price Oscillator
    ppo, ppo_signal, ppo_hist = percentage_price_oscillator(df['close'])
    log(f"PPO: {ppo.iloc[-1]:.4f}, Signal: {ppo_signal.iloc[-1]:.4f}")
    
    # Coppock Curve
    cc = coppock_curve(df['close'])
    log(f"Coppock Curve: {cc.iloc[-1]:.4f}")
    
    # Mass Index
    mi = mass_index(df['high'], df['low'])
    log(f"Mass Index: {mi.iloc[-1]:.4f}")
    
    # Squeeze Momentum
    squeeze_mom, squeeze_on = squeeze_momentum(df['high'], df['low'], df['close'])
    log(f"Squeeze Momentum: {squeeze_mom.iloc[-1]:.4f}, Squeeze On: {squeeze_on.iloc[-1]}")
    
    # Fisher Transform
    fisher, fisher_signal = ehlers_fisher_transform(df['close'])
    log(f"Fisher Transform: {fisher.iloc[-1]:.4f}, Signal: {fisher_signal.iloc[-1]:.4f}")
    
    # Chande Momentum Oscillator
    cmo = chande_momentum_oscillator(df['close'])
    log(f"Chande Momentum Oscillator: {cmo.iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing advanced momentum indicators: {str(e)}")

# Test advanced volume indicators
try:
    log("\n--- Testing Advanced Volume Indicators ---")
    
    # Accumulation Distribution Line
    adl = accumulation_distribution_line(df['high'], df['low'], df['close'], df['volume'])
    log(f"Accumulation Distribution Line: {adl.iloc[-1]:.4f}")
    
    # Chaikin Oscillator
    co = chaikin_oscillator(df['high'], df['low'], df['close'], df['volume'])
    log(f"Chaikin Oscillator: {co.iloc[-1]:.4f}")
    
    # Chaikin Money Flow
    cmf = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
    log(f"Chaikin Money Flow: {cmf.iloc[-1]:.4f}")
    
    # Money Flow Index
    mfi = money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    log(f"Money Flow Index: {mfi.iloc[-1]:.4f}")
    
    # Volume Weighted Average Price
    vwap_val = vwap(df['high'], df['low'], df['close'], df['volume'])
    log(f"VWAP: {vwap_val.iloc[-1]:.4f}")
    
    # Klinger Oscillator
    kvo, kvo_signal = klinger_oscillator(df['high'], df['low'], df['close'], df['volume'])
    log(f"Klinger Oscillator: {kvo.iloc[-1]:.4f}, Signal: {kvo_signal.iloc[-1]:.4f}")
    
    # Volume Profile
    log("\n--- Testing Market Structure and Advanced Analysis ---")
    vp = volume_profile(df['high'], df['low'], df['close'], df['volume'], bins=10, window=50)
    log(f"Volume Profile - POC: {vp['poc']:.4f}, VAH: {vp['vah']:.4f}, VAL: {vp['val']:.4f}")
    log(f"Volume Profile - Number of HVNs: {len(vp['hvn'])}, Number of LVNs: {len(vp['lvn'])}")
    
except Exception as e:
    log(f"Error testing advanced volume indicators: {str(e)}")

# Test harmonic patterns
try:
    # Harmonic Patterns
    hp = harmonic_patterns(df['high'], df['low'], df['close'])
    patterns_found = len(hp['patterns'])
    log(f"Harmonic Patterns - Patterns Found: {patterns_found}")
    if patterns_found > 0:
        for i, pattern in enumerate(hp['patterns']):
            log(f"  Pattern {i+1}: {pattern['type']} with {pattern['confidence']}% confidence")
    
except Exception as e:
    log(f"Error testing harmonic patterns: {str(e)}")

# Test divergence scanner
try:
    # Get RSI for testing divergence
    rsi = relative_strength_index(df['close'])
    
    # Divergence Scanner with RSI
    div = divergence_scanner(df['close'], rsi)
    log(f"Divergence Scanner - Regular Bullish: {div['regular_bullish']}, Regular Bearish: {div['regular_bearish']}")
    log(f"Divergence Scanner - Hidden Bullish: {div['hidden_bullish']}, Hidden Bearish: {div['hidden_bearish']}")
    if div['regular_bullish'] or div['regular_bearish'] or div['hidden_bullish'] or div['hidden_bearish']:
        log(f"Divergence Scanner - Strength: {div['strength']:.2f}%")
    
    # Test with MACD too
    macd_line, _, _ = macd(df['close'])
    div_macd = divergence_scanner(df['close'], macd_line)
    log(f"MACD Divergence - Any Divergence: {any([div_macd['regular_bullish'], div_macd['regular_bearish'], div_macd['hidden_bullish'], div_macd['hidden_bearish']])}")
    
except Exception as e:
    log(f"Error testing divergence scanner: {str(e)}")

# Test newly added indicators
log("\n" + "=" * 50)
log("TESTING NEWLY ADDED INDICATORS")
log("=" * 50)

# Test Stochastic RSI
try:
    log("\n--- Testing Stochastic RSI ---")
    
    # Stochastic RSI
    k, d = stochastic_rsi(df['close'], 
                          rsi_period=14, 
                          stoch_period=14, 
                          k_period=3, 
                          d_period=3)
    log(f"Stochastic RSI - %K: {k.iloc[-1]:.4f}, %D: {d.iloc[-1]:.4f}")
    log(f"Stochastic RSI - Overbought: {k.iloc[-1] > 80}, Oversold: {k.iloc[-1] < 20}")
    
    # Test different periods
    k_short, d_short = stochastic_rsi(df['close'], 
                                    rsi_period=7, 
                                    stoch_period=7, 
                                    k_period=3, 
                                    d_period=3)
    log(f"Short-term Stochastic RSI - %K: {k_short.iloc[-1]:.4f}, %D: {d_short.iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing Stochastic RSI: {str(e)}")

# Test Elliott Wave Tracker
try:
    log("\n--- Testing Elliott Wave Tracker ---")
    
    # Elliott Wave Tracker
    waves = elliott_wave_tracker(df['high'], 
                                 df['low'], 
                                 df['close'], 
                                 df['volume'])
    
    log(f"Elliott Wave Tracker - Wave Count: {waves['wave_count']}")
    log(f"Elliott Wave Tracker - Current Position: {waves['current_position']}")
    log(f"Elliott Wave Tracker - Confidence: {waves['confidence']:.2f}")
    
    # Count of impulse and corrective waves
    impulse_count = len(waves['impulse_waves'])
    corrective_count = len(waves['corrective_waves'])
    log(f"Elliott Wave Tracker - Impulse Waves: {impulse_count}, Corrective Waves: {corrective_count}")
    
    # Print individual waves if available
    if impulse_count > 0:
        log("Impulse Waves:")
        for wave in waves['impulse_waves']:
            log(f"  Wave {wave['wave']} at index {wave['index']} with price {wave['price']:.4f}")
    
    if corrective_count > 0:
        log("Corrective Waves:")
        for wave in waves['corrective_waves']:
            log(f"  Wave {wave['wave']} at index {wave['index']} with price {wave['price']:.4f}")
    
except Exception as e:
    log(f"Error testing Elliott Wave Tracker: {str(e)}")

# Test Mean Reversion Index
try:
    log("\n--- Testing Mean Reversion Index ---")
    
    # Mean Reversion Index
    mri, upper_band, lower_band = mean_reversion_index(df['close'], 
                                                     df['high'], 
                                                     df['low'])
    
    log(f"Mean Reversion Index - MRI: {mri.iloc[-1]:.4f}")
    log(f"Mean Reversion Index - Upper Band: {upper_band.iloc[-1]:.4f}")
    log(f"Mean Reversion Index - Lower Band: {lower_band.iloc[-1]:.4f}")
    
    # Test signals
    buy_signal = mri.iloc[-1] < -50
    sell_signal = mri.iloc[-1] > 50
    log(f"Mean Reversion Index - Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")
    
    # Test different parameters
    mri_custom, ub_custom, lb_custom = mean_reversion_index(df['close'], 
                                                        df['high'], 
                                                        df['low'],
                                                        period=7, 
                                                        std_dev_multiplier=1.5)
    
    log(f"Custom Mean Reversion Index - MRI: {mri_custom.iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing Mean Reversion Index: {str(e)}")

# Test Market Breadth Indicators (if available)
try:
    log("\n--- Testing Market Breadth Indicators ---")
    
    # Create sample market breadth data
    sample_size = len(df)
    advances = pd.Series(np.random.randint(100, 500, sample_size))
    declines = pd.Series(np.random.randint(100, 500, sample_size))
    unchanged = pd.Series(np.random.randint(10, 50, sample_size))
    
    # Market Breadth Indicators
    mb = market_breadth_indicators(advances, declines, unchanged)
    
    log(f"Advance-Decline Line: {mb['advance_decline_line'].iloc[-1]:.4f}")
    log(f"McClellan Oscillator: {mb['mcclellan_oscillator'].iloc[-1]:.4f}")
    log(f"Arms Index (TRIN): {mb['trin'].iloc[-1]:.4f}")
    log(f"Bullish Percent Index: {mb['bullish_percent_index'].iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing Market Breadth Indicators: {str(e)}")

# Test Order Flow Analysis (if available)
try:
    log("\n--- Testing Order Flow Analysis ---")
    
    # Create sample bid/ask volume data
    bid_volume = pd.Series(df['volume'] * (0.4 + 0.3 * np.random.random(len(df))))
    ask_volume = pd.Series(df['volume'] * (0.4 + 0.3 * np.random.random(len(df))))
    
    # Order Flow Analysis
    of = orderflow_analysis(df['close'], df['volume'], bid_volume, ask_volume)
    
    log(f"Delta: {of['delta'].iloc[-1]:.4f}")
    log(f"Cumulative Delta: {of['cumulative_delta'].iloc[-1]:.4f}")
    log(f"Delta Percent: {of['delta_percent'].iloc[-1]:.4f}")
    log(f"Imbalance Ratio: {of['imbalance_ratio'].iloc[-1]:.4f}")
    
except Exception as e:
    log(f"Error testing Order Flow Analysis: {str(e)}")

# Test pivot points and multi-timeframe indicators
try:
    log("\n--- Testing Pivot Points and Multi-Timeframe Indicators ---")
    
    # Camarilla Pivot Points
    cam_pivot, cam_support, cam_resist = camarilla_pivot_points(df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
    log(f"Camarilla - Pivot: {cam_pivot:.4f}, S1: {cam_support['s1']:.4f}, R1: {cam_resist['r1']:.4f}")
    
    # Woodie Pivot Points
    wood_pivot, wood_support, wood_resist = woodie_pivot_points(df['open'].iloc[-1], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
    log(f"Woodie - Pivot: {wood_pivot:.4f}, S1: {wood_support['s1']:.4f}, R1: {wood_resist['r1']:.4f}")
    
    # DeMark Pivot Points
    demark_pivot, demark_support, demark_resist = demark_pivot_points(df['open'].iloc[-1], df['high'].iloc[-2], df['low'].iloc[-2], df['close'].iloc[-2])
    log(f"DeMark - Pivot: {demark_pivot:.4f}, S1: {demark_support['s1']:.4f}, R1: {demark_resist['r1']:.4f}")
    
    # Elder Triple Screen
    if len(weekly_df) >= 14:
        ets = elder_triple_screen(
            df['close'], df['high'], df['low'], df['volume'],
            weekly_df['close'], weekly_df['high'], weekly_df['low'], weekly_df['volume']
        )
        log(f"Elder Triple Screen - Weekly Trend: {ets['weekly_trend']}, Force Index: {ets['force_index'].iloc[-1]:.4f}")
        log(f"Elder Triple Screen - Buy Signal: {ets['buy_signal'].iloc[-1]}, Sell Signal: {ets['sell_signal'].iloc[-1]}")
    else:
        log("Not enough weekly data for Elder Triple Screen test")
    
except Exception as e:
    log(f"Error testing pivot points and multi-timeframe indicators: {str(e)}")

# Test Smart Money Indicators
log("\n" + "=" * 50)
log("TESTING SMART MONEY INDICATORS")
log("=" * 50)

try:
    from app.technical_indicators.smart_money_indicators import (
        liquidity_sweep_analysis, order_block_detection, smart_money_analysis,
        cumulative_delta_analysis, volatility_regime_detection, market_depth_analysis,
        funding_liquidation_analysis, cross_asset_correlation
    )
    log("Successfully imported smart money indicators!")
except Exception as e:
    log(f"Error importing smart money indicators: {str(e)}")

# Test liquidity sweep analysis
try:
    log("\n--- Testing Liquidity Sweep Analysis ---")
    
    sweeps = liquidity_sweep_analysis(df['high'], df['low'], df['close'], df['volume'])
    log(f"Liquidity Sweeps - High Sweeps: {len(sweeps['high_sweeps'])}, Low Sweeps: {len(sweeps['low_sweeps'])}")
    log(f"Current High Sweep: {sweeps['current_high_sweep']}, Current Low Sweep: {sweeps['current_low_sweep']}")
    
    if sweeps['high_sweeps']:
        log(f"Sample High Sweep - Price: {sweeps['high_sweeps'][0]['price']:.4f}, Strength: {sweeps['high_sweeps'][0]['strength']:.2f}")
    
    if sweeps['low_sweeps']:
        log(f"Sample Low Sweep - Price: {sweeps['low_sweeps'][0]['price']:.4f}, Strength: {sweeps['low_sweeps'][0]['strength']:.2f}")
    
except Exception as e:
    log(f"Error testing liquidity sweep analysis: {str(e)}")

# Test order block detection
try:
    log("\n--- Testing Order Block Detection ---")
    
    order_blocks = order_block_detection(df['open'], df['high'], df['low'], df['close'], df['volume'])
    log(f"Order Blocks - Bullish: {len(order_blocks['bullish_order_blocks'])}, Bearish: {len(order_blocks['bearish_order_blocks'])}")
    log(f"Active Blocks - Bullish: {len(order_blocks['active_bullish_blocks'])}, Bearish: {len(order_blocks['active_bearish_blocks'])}")
    
    if order_blocks['bullish_order_blocks']:
        log(f"Sample Bullish Block - Low: {order_blocks['bullish_order_blocks'][0]['low']:.4f}, Strength: {order_blocks['bullish_order_blocks'][0]['strength']:.2f}")
    
    if order_blocks['bearish_order_blocks']:
        log(f"Sample Bearish Block - High: {order_blocks['bearish_order_blocks'][0]['high']:.4f}, Strength: {order_blocks['bearish_order_blocks'][0]['strength']:.2f}")
    
except Exception as e:
    log(f"Error testing order block detection: {str(e)}")

# Test smart money analysis
try:
    log("\n--- Testing Smart Money Analysis ---")
    
    smc = smart_money_analysis(df['open'], df['high'], df['low'], df['close'], df['volume'])
    log(f"Fair Value Gaps - Bullish: {len(smc['bullish_fvg'])}, Bearish: {len(smc['bearish_fvg'])}")
    log(f"Equal Levels - Highs: {len(smc['equal_highs'])}, Lows: {len(smc['equal_lows'])}")
    
    if smc['bullish_fvg']:
        log(f"Sample Bullish FVG - Size: {smc['bullish_fvg'][0]['size']:.4f}, Mitigated: {smc['bullish_fvg'][0]['mitigated']}")
    
    if smc['equal_highs']:
        log(f"Sample Equal High - Price: {smc['equal_highs'][0]['price']:.4f}, Swept: {smc['equal_highs'][0]['swept']}")
    
except Exception as e:
    log(f"Error testing smart money analysis: {str(e)}")

# Test cumulative delta analysis
try:
    log("\n--- Testing Cumulative Delta Analysis ---")
    
    delta = cumulative_delta_analysis(df['open'], df['high'], df['low'], df['close'], df['volume'])
    log(f"Cumulative Delta - Divergences: {len(delta['delta_divergences'])}")
    
    # Check first and last values of delta
    if len(delta['delta']) > 0:
        log(f"Delta First: {delta['delta'][0]:.2f}, Last: {delta['delta'][-1]:.2f}")
    
    # Check first and last values of cumulative delta
    if len(delta['cumulative_delta']) > 0:
        log(f"Cumulative Delta First: {delta['cumulative_delta'][0]:.2f}, Last: {delta['cumulative_delta'][-1]:.2f}")
    
    # Check a few divergences if available
    if delta['delta_divergences']:
        log(f"Sample Divergence - Type: {delta['delta_divergences'][0]['type']}, Price: {delta['delta_divergences'][0]['price']:.4f}")
    
except Exception as e:
    log(f"Error testing cumulative delta analysis: {str(e)}")

# Test volatility regime detection
try:
    log("\n--- Testing Volatility Regime Detection ---")
    
    vol_regime = volatility_regime_detection(df['close'], df['high'], df['low'])
    log(f"Volatility Regime: {vol_regime['regime']}")
    log(f"Volatility Percentile: {vol_regime['volatility_percentile']}")
    log(f"Volatility Ratio: {vol_regime['volatility_ratio']:.4f}")
    log(f"Short ATR: {vol_regime['short_atr']:.4f}, Long ATR: {vol_regime['long_atr']:.4f}")
    
    # Check indicator adjustments based on regime
    log(f"Adjusted RSI Period: {vol_regime['indicator_adjustments']['rsi_period']}")
    log(f"Adjusted MACD Fast Period: {vol_regime['indicator_adjustments']['macd_fast']}")
    log(f"Adjusted ATR Multiplier: {vol_regime['indicator_adjustments']['atr_multiplier']:.2f}")
    
except Exception as e:
    log(f"Error testing volatility regime detection: {str(e)}")

# Test market depth analysis with mock data
try:
    log("\n--- Testing Market Depth Analysis ---")
    
    # Create mock order book data
    current_price = 100.0
    bid_levels = [99.9, 99.8, 99.5, 99.0, 98.5, 98.0, 97.0, 96.0, 95.0]
    ask_levels = [100.1, 100.2, 100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0]
    volumes = [10, 15, 25, 50, 100, 75, 25, 10, 5]
    
    depth = market_depth_analysis(current_price, bid_levels, ask_levels, volumes)
    log(f"Market Depth - Bid/Ask Imbalance: {depth['bid_ask_imbalance']:.4f}")
    log(f"Buy Pressure: {depth['buy_pressure']:.2f}, Sell Pressure: {depth['sell_pressure']:.2f}")
    log(f"Support Clusters: {len(depth['support_clusters'])}, Resistance Clusters: {len(depth['resistance_clusters'])}")
    
    if depth['support_clusters']:
        log(f"Sample Support Cluster - Price: {depth['support_clusters'][0]['price']:.4f}, Relative Size: {depth['support_clusters'][0]['relative_size']:.4f}")
    
    if depth['resistance_clusters']:
        log(f"Sample Resistance Cluster - Price: {depth['resistance_clusters'][0]['price']:.4f}, Relative Size: {depth['resistance_clusters'][0]['relative_size']:.4f}")
    
except Exception as e:
    log(f"Error testing market depth analysis: {str(e)}")

# Test funding rate analysis with mock data
try:
    log("\n--- Testing Funding Rate Analysis ---")
    
    # Create mock funding and liquidation data
    current_price = 100.0
    dates = pd.date_range(start='2022-01-01', periods=30)
    
    # Create sample funding rates (fluctuating between positive and negative)
    funding_rate = pd.Series([
        0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0000, -0.0001, -0.0002, -0.0001, 0.0000,
        0.0001, 0.0003, 0.0005, 0.0004, 0.0003, 0.0001, 0.0000, -0.0001, -0.0003, -0.0005,
        -0.0003, -0.0001, 0.0000, 0.0001, 0.0002, 0.0001, 0.0000, -0.0001, -0.0002, -0.0001
    ], index=dates)
    
    # Create sample liquidation data
    long_liquidations = pd.Series([
        100, 200, 150, 300, 500, 1200, 300, 200, 150, 100,
        200, 150, 100, 200, 300, 500, 1500, 300, 200, 150,
        100, 200, 300, 250, 150, 100, 200, 150, 100, 200
    ], index=dates)
    
    short_liquidations = pd.Series([
        150, 250, 200, 150, 300, 200, 1000, 600, 300, 200,
        150, 100, 200, 150, 300, 200, 100, 1200, 500, 300,
        200, 150, 100, 200, 150, 300, 200, 150, 100, 200
    ], index=dates)
    
    # Create sample open interest data
    open_interest = pd.Series([
        10000, 10200, 10400, 10500, 10300, 10100, 10200, 10400, 10600, 10800,
        11000, 11200, 11500, 11800, 12000, 12200, 12100, 11900, 11700, 11500,
        11300, 11100, 11000, 10900, 10800, 10700, 10900, 11100, 11300, 11500
    ], index=dates)
    
    funding = funding_liquidation_analysis(current_price, funding_rate, long_liquidations, short_liquidations, open_interest)
    log(f"Funding Analysis - Trend: {funding['funding_trend']}")
    log(f"Current Funding Rate: {funding['current_funding']:.6f}, Avg Funding: {funding['avg_funding']:.6f}")
    log(f"Open Interest Change: {funding['open_interest_change']:.2f}%")
    log(f"Market Sentiment: {funding['market_sentiment']}")
    log(f"Long Liquidation Clusters: {len(funding['long_liquidation_clusters'])}")
    log(f"Short Liquidation Clusters: {len(funding['short_liquidation_clusters'])}")
    
except Exception as e:
    log(f"Error testing funding rate analysis: {str(e)}")

# Test cross-asset correlation with mock data
try:
    log("\n--- Testing Cross-Asset Correlation ---")
    
    # Create mock price data for main asset and related assets
    dates = pd.date_range(start='2022-01-01', periods=100)
    
    # Main asset (e.g., Bitcoin)
    main_asset = pd.Series(np.cumsum(np.random.normal(0.002, 0.02, 100)), index=dates)
    
    # Related assets (e.g., Ethereum, S&P 500, Gold)
    related_assets = {
        "ETH": pd.Series(np.cumsum(np.random.normal(0.002, 0.025, 100) + 0.0005 * main_asset.values), index=dates),  # Positively correlated
        "SPX": pd.Series(np.cumsum(np.random.normal(0.001, 0.01, 100) + 0.0001 * main_asset.values), index=dates),   # Weakly correlated
        "GOLD": pd.Series(np.cumsum(np.random.normal(0.0005, 0.008, 100) - 0.0002 * main_asset.values), index=dates) # Negatively correlated
    }
    
    corr = cross_asset_correlation(main_asset, related_assets)
    
    log(f"Strongest Positive Correlation: {corr['strongest_positive']['asset']} ({corr['strongest_positive']['correlation']:.4f})")
    log(f"Strongest Negative Correlation: {corr['strongest_negative']['asset']} ({corr['strongest_negative']['correlation']:.4f})")
    
    # Check correlations by timeframe
    for period in ['14d', '30d', '90d']:
        if period in corr['correlations']:
            log(f"\nCorrelations for {period}:")
            for asset, data in corr['correlations'][period].items():
                log(f"  - {asset}: {data['correlation']:.4f} ({data['strength']})")
    
except Exception as e:
    log(f"Error testing cross-asset correlation: {str(e)}")

# Test Complete
log("\n" + "=" * 50)
log("TESTING COMPLETE")
log("=" * 50)
log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("\nTests completed and results saved to indicator_test_log.txt")

# Close the log file
log_file.close() 