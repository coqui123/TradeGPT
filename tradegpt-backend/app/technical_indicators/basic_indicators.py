"""
Basic Technical Indicators
Contains commonly used technical indicators
"""
import pandas as pd
import numpy as np

def macd(close, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
 
def bollinger_bands(close, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def relative_strength_index(close, window=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def average_true_range(high, low, close, window=14):
    """Calculate ATR (Average True Range)"""
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def pivot_points(high, low, close):
    """Calculate Pivot Points"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def elder_ray_index(high, low, close, window=13):
    """Calculate Elder Ray Index"""
    ema = close.ewm(span=window, adjust=False).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def stochastic_oscillator(high, low, close, window=14, smooth_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=smooth_window).mean()
    return k, d

def on_balance_volume(close, volume):
    """Calculate On Balance Volume (OBV)"""
    return (np.sign(close.diff()) * volume).cumsum()

def keltner_channel(high, low, close, window=20, atr_window=10, multiplier=2):
    """Calculate Keltner Channels"""
    typical_price = (high + low + close) / 3
    atr_val = average_true_range(high, low, close, window=atr_window)
    middle = typical_price.rolling(window=window).mean()
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower

def directional_movement_index(high, low, close, window=14):
    """Calculate DMI (Directional Movement Index)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    plus_di = 100 * (plus_dm.rolling(window).sum() / atr)
    minus_di = abs(100 * (minus_dm.rolling(window).sum() / atr))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()

    return plus_di, minus_di, adx 