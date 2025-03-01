"""
Advanced Technical Indicators
Contains more specialized technical indicators
"""
import pandas as pd
import numpy as np
from app.technical_indicators.basic_indicators import average_true_range

def accumulation_distribution_line(high, low, close, volume):
    """Calculate Accumulation Distribution Line (ADL)"""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0)  # Handle division by zero
    adl = (clv * volume).cumsum()
    return adl

def chaikin_oscillator(high, low, close, volume, short_period=3, long_period=10):
    """Calculate Chaikin Oscillator"""
    adl = accumulation_distribution_line(high, low, close, volume)
    return adl.ewm(span=short_period, adjust=False).mean() - adl.ewm(span=long_period, adjust=False).mean()

def aroon_indicator(high, low, window=14):
    """Calculate Aroon Indicator"""
    high_idx = high.rolling(window=window).apply(lambda x: x.argmax())
    low_idx = low.rolling(window=window).apply(lambda x: x.argmin())
    
    aroon_up = ((window - high_idx) / window) * 100
    aroon_down = ((window - low_idx) / window) * 100
    return aroon_up, aroon_down

def chaikin_money_flow(high, low, close, volume, window=20):
    """Calculate Chaikin Money Flow"""
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    money_flow_volume = money_flow_multiplier * volume
    return money_flow_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()

def parabolic_sar(high, low, step=0.02, max_step=0.2):
    """Calculate Parabolic SAR"""
    # Initialize variables
    sar = pd.Series(index=high.index)
    trend = pd.Series(index=high.index)
    extreme_point = pd.Series(index=high.index)
    acceleration_factor = pd.Series(index=high.index)
    
    # Set initial values
    trend.iloc[0] = 1  # Start with uptrend (1 for uptrend, -1 for downtrend)
    sar.iloc[0] = low.iloc[0]
    extreme_point.iloc[0] = high.iloc[0]
    acceleration_factor.iloc[0] = step
    
    # Calculate SAR values
    for i in range(1, len(high)):
        # Previous SAR value
        prev_sar = sar.iloc[i-1]
        
        # Current SAR value based on previous trend
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = prev_sar + acceleration_factor.iloc[i-1] * (extreme_point.iloc[i-1] - prev_sar)
            # Ensure SAR is below the previous two lows
            sar.iloc[i] = min(sar.iloc[i], low.iloc[max(0, i-2):i].min())
            
            # Check for trend reversal
            if sar.iloc[i] > low.iloc[i]:
                trend.iloc[i] = -1  # Change to downtrend
                sar.iloc[i] = extreme_point.iloc[i-1]
                extreme_point.iloc[i] = low.iloc[i]
                acceleration_factor.iloc[i] = step
            else:
                trend.iloc[i] = 1  # Continue uptrend
                extreme_point.iloc[i] = max(extreme_point.iloc[i-1], high.iloc[i])
                acceleration_factor.iloc[i] = min(max_step, acceleration_factor.iloc[i-1] + step) if extreme_point.iloc[i] > extreme_point.iloc[i-1] else acceleration_factor.iloc[i-1]
        else:  # Downtrend
            sar.iloc[i] = prev_sar - acceleration_factor.iloc[i-1] * (prev_sar - extreme_point.iloc[i-1])
            # Ensure SAR is above the previous two highs
            sar.iloc[i] = max(sar.iloc[i], high.iloc[max(0, i-2):i].max())
            
            # Check for trend reversal
            if sar.iloc[i] < high.iloc[i]:
                trend.iloc[i] = 1  # Change to uptrend
                sar.iloc[i] = extreme_point.iloc[i-1]
                extreme_point.iloc[i] = high.iloc[i]
                acceleration_factor.iloc[i] = step
            else:
                trend.iloc[i] = -1  # Continue downtrend
                extreme_point.iloc[i] = min(extreme_point.iloc[i-1], low.iloc[i])
                acceleration_factor.iloc[i] = min(max_step, acceleration_factor.iloc[i-1] + step) if extreme_point.iloc[i] < extreme_point.iloc[i-1] else acceleration_factor.iloc[i-1]
    
    return sar, trend

def money_flow_index(high, low, close, volume, window=14):
    """Calculate Money Flow Index"""
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate raw money flow
    raw_money_flow = typical_price * volume
    
    # Get the direction of money flow (positive or negative)
    direction = np.where(typical_price > typical_price.shift(1), 1, -1)
    direction[0] = 0  # Set first value to neutral
    
    # Calculate positive and negative money flow
    positive_flow = np.where(direction > 0, raw_money_flow, 0)
    negative_flow = np.where(direction < 0, raw_money_flow, 0)
    
    # Calculate money flow ratio
    positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=window).sum()
    
    # Calculate Money Flow Index
    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def percentage_price_oscillator(close, fast_period=12, slow_period=26, signal_period=9):
    """Calculate Percentage Price Oscillator"""
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100
    signal_line = ppo_line.ewm(span=signal_period, adjust=False).mean()
    histogram = ppo_line - signal_line
    return ppo_line, signal_line, histogram

def donchian_channels(high, low, window=20):
    """Calculate Donchian Channels"""
    upper = high.rolling(window=window).max()
    lower = low.rolling(window=window).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def rate_of_change(close, window=14):
    """Calculate Rate of Change"""
    return (close / close.shift(window) - 1) * 100

def commodity_channel_index(high, low, close, window=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    mean_dev = abs(typical_price - typical_price.rolling(window=window).mean()).rolling(window=window).mean()
    cci = (typical_price - typical_price.rolling(window=window).mean()) / (0.015 * mean_dev)
    return cci

def awesome_oscillator(high, low, short_period=5, long_period=34):
    """Calculate Awesome Oscillator"""
    median_price = (high + low) / 2
    ao = median_price.rolling(window=short_period).mean() - median_price.rolling(window=long_period).mean()
    return ao

def vortex_indicator(high, low, close, period=14):
    """Calculate Vortex Indicator"""
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    plus_vm = abs(high - low.shift())
    minus_vm = abs(low - high.shift())
    
    plus_vi = plus_vm.rolling(window=period).sum() / tr.rolling(window=period).sum()
    minus_vi = minus_vm.rolling(window=period).sum() / tr.rolling(window=period).sum()
    
    return plus_vi, minus_vi

def true_strength_index(close, r=25, s=13):
    """Calculate True Strength Index"""
    momentum = close.diff()
    # Double smoothed momentum
    smoothed1 = momentum.ewm(span=r, adjust=False).mean()
    smoothed2 = smoothed1.ewm(span=s, adjust=False).mean()
    # Double smoothed absolute momentum
    abs_smoothed1 = abs(momentum).ewm(span=r, adjust=False).mean()
    abs_smoothed2 = abs_smoothed1.ewm(span=s, adjust=False).mean()
    # TSI
    tsi = 100 * (smoothed2 / abs_smoothed2)
    return tsi

def mass_index(high, low, period=9, period2=25):
    """Calculate Mass Index"""
    range_ema1 = (high - low).ewm(span=period, adjust=False).mean()
    range_ema2 = range_ema1.ewm(span=period, adjust=False).mean()
    ratio = range_ema1 / range_ema2
    return ratio.rolling(window=period2).sum()

def hull_moving_average(close, period=14):
    """Calculate Hull Moving Average"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = close.rolling(window=half_period).apply(lambda x: pd.Series(x).ewm(span=half_period).mean()[-1], raw=True)
    wma2 = close.rolling(window=period).apply(lambda x: pd.Series(x).ewm(span=period).mean()[-1], raw=True)
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.rolling(window=sqrt_period).apply(lambda x: pd.Series(x).ewm(span=sqrt_period).mean()[-1], raw=True)
    return hma

def coppock_curve(close, roc1=14, roc2=11, period=10):
    """Calculate Coppock Curve"""
    roc1_val = (close / close.shift(roc1) - 1) * 100
    roc2_val = (close / close.shift(roc2) - 1) * 100
    return (roc1_val + roc2_val).ewm(span=period, adjust=False).mean()

def vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def klinger_oscillator(high, low, close, volume, short_period=34, long_period=55, signal_period=13):
    """
    Calculate Klinger Oscillator
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    volume : pd.Series
        Series of volume
    short_period : int
        Short period
    long_period : int
        Long period
    signal_period : int
        Signal period
        
    Returns:
    --------
    pd.Series, pd.Series
        Klinger Oscillator and Signal Line
    """
    # Calculate the typical price
    tp = (high + low + close) / 3
    
    # Calculate the volume force
    vf = volume * np.abs(tp - tp.shift(1)) * np.where(tp > tp.shift(1), 1, -1)
    
    # Calculate the short and long EMAs of volume force
    vf_ema_short = vf.ewm(span=short_period, adjust=False).mean()
    vf_ema_long = vf.ewm(span=long_period, adjust=False).mean()
    
    # Calculate the Klinger Oscillator
    kvo = vf_ema_short - vf_ema_long
    
    # Calculate the signal line
    signal = kvo.ewm(span=signal_period, adjust=False).mean()
    
    return kvo, signal 