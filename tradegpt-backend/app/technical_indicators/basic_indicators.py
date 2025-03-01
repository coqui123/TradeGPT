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

def williams_percent_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr

def volume_zone_oscillator(close, volume, period=14, signal_period=9):
    """Calculate Volume Zone Oscillator"""
    price_change = close.diff()
    pos_volume = np.where(price_change > 0, volume, 0)
    neg_volume = np.where(price_change < 0, volume, 0)
    zero_volume = np.where(price_change == 0, volume, 0)
    
    pos_volume_sma = pd.Series(pos_volume).rolling(window=period).sum()
    neg_volume_sma = pd.Series(neg_volume).rolling(window=period).sum()
    zero_volume_sma = pd.Series(zero_volume).rolling(window=period).sum()
    
    total_volume_sma = pos_volume_sma + neg_volume_sma + zero_volume_sma
    
    vzo = 100 * (pos_volume_sma - neg_volume_sma) / total_volume_sma
    signal = vzo.rolling(window=signal_period).mean()
    
    return vzo, signal

def relative_volume(volume, window=20):
    """Calculate Relative Volume (comparing current volume to average)"""
    avg_volume = volume.rolling(window=window).mean()
    rel_volume = volume / avg_volume
    return rel_volume

def chandelier_exit(high, low, close, period=22, atr_multiplier=3.0):
    """Calculate Chandelier Exit"""
    atr = average_true_range(high, low, close, window=period)
    
    # Long exit (for uptrend)
    highest_high = high.rolling(window=period).max()
    long_exit = highest_high - (atr * atr_multiplier)
    
    # Short exit (for downtrend)
    lowest_low = low.rolling(window=period).min()
    short_exit = lowest_low + (atr * atr_multiplier)
    
    return long_exit, short_exit

def market_regime(close, sma_fast=50, sma_slow=200, std_period=20, std_threshold=1.5):
    """Identify market regime (trending vs ranging)"""
    # Calculate SMAs
    sma_f = close.rolling(window=sma_fast).mean()
    sma_s = close.rolling(window=sma_slow).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=std_period).std()
    std_avg = std.rolling(window=std_period).mean()
    
    # Determine volatility regime
    volatility_regime = std > (std_avg * std_threshold)
    
    # Determine trend regime (1: uptrend, -1: downtrend, 0: ranging)
    trend_regime = np.zeros(len(close))
    trend_regime[sma_f > sma_s] = 1  # Uptrend
    trend_regime[sma_f < sma_s] = -1  # Downtrend
    
    # Combine to get market regime
    # 1: volatile uptrend, -1: volatile downtrend
    # 0.5: low-volatility uptrend, -0.5: low-volatility downtrend
    # 0: ranging/uncertain
    market_regime = pd.Series(trend_regime, index=close.index)
    market_regime[volatility_regime & (trend_regime == 1)] = 1
    market_regime[volatility_regime & (trend_regime == -1)] = -1
    market_regime[~volatility_regime & (trend_regime == 1)] = 0.5
    market_regime[~volatility_regime & (trend_regime == -1)] = -0.5
    
    return market_regime, volatility_regime

