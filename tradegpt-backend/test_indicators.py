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
        volume_profile, harmonic_patterns, divergence_scanner
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

# Test Complete
log("\n" + "=" * 50)
log("TESTING COMPLETE")
log("=" * 50)
log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("\nTests completed and results saved to indicator_test_log.txt")

# Close the log file
log_file.close() 