"""
Advanced Technical Indicators
Contains more specialized technical indicators
"""
import pandas as pd
import numpy as np
from app.technical_indicators.basic_indicators import average_true_range, bollinger_bands, macd, relative_strength_index

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
    # Calculate WMA with period/2
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # Use simple pandas EWM instead of the custom lambda
    wma1 = close.ewm(span=half_period, adjust=False).mean()
    wma2 = close.ewm(span=period, adjust=False).mean()
    
    # Calculate 2 * WMA(n/2) - WMA(n)
    raw_hma = 2 * wma1 - wma2
    
    # Calculate WMA with sqrt(n)
    hma = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
    
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

def ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    Calculate Ichimoku Cloud components
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    tenkan_period : int
        Period for Tenkan-sen (Conversion Line)
    kijun_period : int
        Period for Kijun-sen (Base Line)
    senkou_b_period : int
        Period for Senkou Span B (Leading Span B)
    displacement : int
        Displacement period for Kumo (Cloud)
        
    Returns:
    --------
    pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
        Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span
    """
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
    tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                 low.rolling(window=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
    kijun_sen = (high.rolling(window=kijun_period).max() + 
                low.rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods, displaced forward 26 periods
    senkou_span_b = ((high.rolling(window=senkou_b_period).max() + 
                    low.rolling(window=senkou_b_period).min()) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Current closing price displaced backwards 26 periods
    chikou_span = close.shift(-displacement)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend Indicator
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    period : int
        ATR period
    multiplier : float
        ATR multiplier
        
    Returns:
    --------
    pd.Series, pd.Series
        Supertrend values, Direction (1: uptrend, -1: downtrend)
    """
    # Calculate ATR
    atr = average_true_range(high, low, close, window=period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize SuperTrend and direction series
    supertrend = pd.Series(0.0, index=close.index)
    direction = pd.Series(0, index=close.index)
    
    # Set initial values
    supertrend.iloc[period-1] = lower_band.iloc[period-1]  # Assume uptrend initially
    direction.iloc[period-1] = 1
    
    # Calculate SuperTrend values
    for i in range(period, len(close)):
        # Current close crosses above upper band - Trend changes to down
        if close.iloc[i-1] <= upper_band.iloc[i-1] and close.iloc[i] > upper_band.iloc[i]:
            direction.iloc[i] = -1
        
        # Current close crosses below lower band - Trend changes to up
        elif close.iloc[i-1] >= lower_band.iloc[i-1] and close.iloc[i] < lower_band.iloc[i]:
            direction.iloc[i] = 1
        
        # No trend change
        else:
            direction.iloc[i] = direction.iloc[i-1]
            
            # Adjust bands based on direction
            if direction.iloc[i] == 1 and lower_band.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            elif direction.iloc[i] == -1 and upper_band.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            else:
                supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
                
    return supertrend, direction

def heikin_ashi(open_price, high, low, close):
    """
    Calculate Heikin-Ashi candles for smoother trend visualization
    
    Parameters:
    -----------
    open_price : pd.Series
        Series of open prices
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
        
    Returns:
    --------
    pd.Series, pd.Series, pd.Series, pd.Series
        Heikin-Ashi open, high, low, close values
    """
    ha_close = (open_price + high + low + close) / 4
    
    # Initialize with first candle
    ha_open = pd.Series(index=open_price.index)
    ha_open.iloc[0] = (open_price.iloc[0] + close.iloc[0]) / 2
    
    # Calculate remaining ha_open values
    for i in range(1, len(close)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    
    return ha_open, ha_high, ha_low, ha_close

def camarilla_pivot_points(high, low, close):
    """
    Calculate Camarilla Pivot Points which provide tighter support/resistance levels
    
    Parameters:
    -----------
    high : float
        Previous period's high
    low : float
        Previous period's low
    close : float
        Previous period's close
        
    Returns:
    --------
    float, dict, dict
        Pivot point, Support levels, Resistance levels
    """
    pivot = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla equations
    r4 = close + range_val * 1.1/2
    r3 = close + range_val * 1.1/4
    r2 = close + range_val * 1.1/6
    r1 = close + range_val * 1.1/12
    
    s1 = close - range_val * 1.1/12
    s2 = close - range_val * 1.1/6
    s3 = close - range_val * 1.1/4
    s4 = close - range_val * 1.1/2
    
    return pivot, {'s1': s1, 's2': s2, 's3': s3, 's4': s4}, {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4}

def woodie_pivot_points(open_price, high, low, close):
    """
    Calculate Woodie Pivot Points which emphasize the current open price
    
    Parameters:
    -----------
    open_price : float
        Current period's opening price
    high : float
        Previous period's high
    low : float
        Previous period's low
    close : float
        Previous period's close
        
    Returns:
    --------
    float, dict, dict
        Pivot point, Support levels, Resistance levels
    """
    pivot = (high + low + 2*close) / 4  # Weighted with more emphasis on close
    r2 = pivot + (high - low)
    r1 = 2*pivot - low
    s1 = 2*pivot - high
    s2 = pivot - (high - low)
    
    # Woodie's R3, R4, S3, S4 calculations
    r3 = high + 2 * (pivot - low)
    r4 = r3 + (high - low)
    s3 = low - 2 * (high - pivot)
    s4 = s3 - (high - low)
    
    return pivot, {'s1': s1, 's2': s2, 's3': s3, 's4': s4}, {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4}

def demark_pivot_points(open_price, high, low, close):
    """
    Calculate DeMark Pivot Points which use conditional formulas based on close vs open relationship
    
    Parameters:
    -----------
    open_price : float
        Current period's opening price
    high : float
        Previous period's high
    low : float
        Previous period's low
    close : float
        Previous period's close
        
    Returns:
    --------
    float, dict, dict
        Pivot point, Support levels, Resistance levels
    """
    # Determine X based on close-open relationship
    if close < open_price:
        x = high + 2*low + close
    elif close > open_price:
        x = 2*high + low + close
    else:
        x = high + low + 2*close
    
    pivot = x / 4
    
    # DeMark support and resistance
    r1 = x / 2 - low
    s1 = x / 2 - high
    
    # Extended levels
    r2 = pivot + (r1 - pivot)
    s2 = pivot - (pivot - s1)
    
    return pivot, {'s1': s1, 's2': s2}, {'r1': r1, 'r2': r2}

def squeeze_momentum(high, low, close, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
    """
    Calculate Squeeze Momentum Indicator (John Carter)
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    bb_length : int
        Bollinger Bands period
    bb_mult : float
        Bollinger Bands standard deviation multiplier
    kc_length : int
        Keltner Channel period
    kc_mult : float
        Keltner Channel ATR multiplier
        
    Returns:
    --------
    pd.Series, pd.Series
        Momentum, Squeeze On/Off indicator (1: on, 0: off)
    """
    # Calculate Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, window=bb_length, num_std=bb_mult)
    
    # Calculate Keltner Channels
    atr = average_true_range(high, low, close, window=kc_length)
    kc_mid = close.rolling(window=kc_length).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr
    
    # Determine if squeeze is on
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Calculate momentum
    highest_high = high.rolling(window=kc_length).max()
    lowest_low = low.rolling(window=kc_length).min()
    m = close - (highest_high + lowest_low) / 2
    
    # Normalize momentum based on True Range
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    avg_tr = tr.rolling(window=kc_length).mean()
    momentum = m / avg_tr
    
    return momentum, squeeze_on.astype(int)

def ehlers_fisher_transform(close, period=10):
    """
    Calculate Ehlers Fisher Transform
    
    Parameters:
    -----------
    close : pd.Series
        Series of close prices
    period : int
        Lookback period
        
    Returns:
    --------
    pd.Series, pd.Series
        Fisher Transform value, Signal line (1-period EMA of Fisher)
    """
    # Calculate the Midpoint Price for the period
    median_price = close.rolling(window=period).median()
    
    # Calculate highest high and lowest low for the period
    highest_high = close.rolling(window=period).max()
    lowest_low = close.rolling(window=period).min()
    
    # Calculate raw value
    raw_value = pd.Series(0.0, index=close.index)
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    # Where we have valid range, compute the normalized price
    valid_range = denominator != 0
    raw_value[valid_range] = ((median_price - lowest_low) / (highest_high - lowest_low) - 0.5) * 2
    
    # Ensure values are within -0.999 to 0.999 for Fisher Transform
    raw_value = raw_value.clip(-0.999, 0.999)
    
    # Apply Fisher Transform
    fisher = 0.5 * np.log((1 + raw_value) / (1 - raw_value))
    
    # Signal line is a 1-period EMA of Fisher
    signal = fisher.ewm(span=1, adjust=False).mean()
    
    return fisher, signal

def chande_momentum_oscillator(close, period=14):
    """
    Calculate Chande Momentum Oscillator
    
    Parameters:
    -----------
    close : pd.Series
        Series of close prices
    period : int
        Lookback period
        
    Returns:
    --------
    pd.Series
        CMO values
    """
    # Get price changes
    price_changes = close.diff()
    
    # Sum of gains and losses in the period
    gains = price_changes.copy()
    losses = price_changes.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = losses.abs()
    
    # Rolling sum of gains and losses
    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()
    
    # Calculate CMO: 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    
    return cmo

def elder_triple_screen(close, high, low, volume, weekly_close, weekly_high, weekly_low, weekly_volume, 
                         impulse_period=13, histogram_period=12, force_period=13):
    """
    Calculate Elder Triple Screen components
    
    Parameters:
    -----------
    close, high, low, volume : pd.Series
        Daily price/volume data
    weekly_close, weekly_high, weekly_low, weekly_volume : pd.Series
        Weekly price/volume data (longer timeframe)
    impulse_period, histogram_period, force_period : int
        Periods for different components
        
    Returns:
    --------
    dict
        Dictionary containing all Elder Triple Screen components
    """
    # 1. Trend identification (weekly timeframe)
    weekly_macd_line, weekly_signal, weekly_histogram = macd(weekly_close)
    weekly_rsi = relative_strength_index(weekly_close, window=14)
    
    # 2. Entry timing (daily timeframe)
    daily_force_index = (close - close.shift(1)) * volume
    daily_force_index_ema = daily_force_index.ewm(span=force_period, adjust=False).mean()
    
    # 3. Impulse system
    ema1 = close.ewm(span=impulse_period, adjust=False).mean()
    ema2 = close.ewm(span=2*impulse_period, adjust=False).mean()
    
    # MACD Histogram for Impulse
    _, _, daily_histogram = macd(close, fast_period=histogram_period, 
                                slow_period=histogram_period*2, signal_period=9)
    
    # Color coding (1: green, -1: red, 0: blue)
    price_color = pd.Series(0, index=close.index)
    price_color[(ema1 > ema1.shift(1)) & (ema1 > ema2)] = 1
    price_color[(ema1 < ema1.shift(1)) & (ema1 < ema2)] = -1
    
    histogram_color = pd.Series(0, index=close.index)
    histogram_color[daily_histogram > daily_histogram.shift(1)] = 1
    histogram_color[daily_histogram < daily_histogram.shift(1)] = -1
    
    # Impulse (2: strong buy, 1: buy, 0: neutral, -1: sell, -2: strong sell)
    impulse = price_color + histogram_color
    
    # 4. Buy/Sell signals
    buy_signal = (weekly_macd_line > weekly_signal) & (impulse > 0) & (daily_force_index_ema > 0)
    sell_signal = (weekly_macd_line < weekly_signal) & (impulse < 0) & (daily_force_index_ema < 0)
    
    return {
        'impulse': impulse,
        'weekly_trend': 1 if weekly_macd_line.iloc[-1] > weekly_signal.iloc[-1] else -1,
        'weekly_rsi': weekly_rsi,
        'force_index': daily_force_index_ema,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'price_color': price_color,
        'histogram_color': histogram_color
    }

def volume_profile(high, low, close, volume, bins=10, window=30):
    """
    Calculate Volume Profile to identify key value areas
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    volume : pd.Series
        Series of volume data
    bins : int
        Number of price bins to divide the range into
    window : int
        Number of periods to look back
        
    Returns:
    --------
    dict
        Dictionary containing POC (Point of Control), VAH (Value Area High),
        VAL (Value Area Value), and HVN/LVN (High/Low Volume Nodes)
    """
    if len(close) < window:
        return {
            "poc": None,
            "vah": None,
            "val": None,
            "hvn": [],
            "lvn": []
        }
    
    # Get the last window periods
    high_window = high[-window:].copy()
    low_window = low[-window:].copy()
    close_window = close[-window:].copy()
    volume_window = volume[-window:].copy()
    
    # Define price range
    price_high = high_window.max()
    price_low = low_window.min()
    price_range = price_high - price_low
    
    # Create bins
    bin_size = price_range / bins
    bin_edges = [price_low + i * bin_size for i in range(bins + 1)]
    
    # Initialize volume array for each bin
    bin_volumes = [0] * bins
    
    # Calculate typical price for each period
    typical_price = (high_window + low_window + close_window) / 3
    
    # Distribute volume across bins
    for i in range(len(typical_price)):
        price = typical_price.iloc[i]
        vol = volume_window.iloc[i]
        
        # Find which bin this price belongs to
        for j in range(bins):
            if bin_edges[j] <= price < bin_edges[j + 1]:
                bin_volumes[j] += vol
                break
    
    # Find Point of Control (POC) - price level with highest volume
    poc_bin = bin_volumes.index(max(bin_volumes))
    poc = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2
    
    # Sort bins by volume
    sorted_bins = sorted(range(bins), key=lambda i: bin_volumes[i], reverse=True)
    
    # Find Value Area (70% of total volume)
    total_volume = sum(bin_volumes)
    target_volume = total_volume * 0.7
    cumulative_volume = 0
    value_area_bins = []
    
    for bin_idx in sorted_bins:
        value_area_bins.append(bin_idx)
        cumulative_volume += bin_volumes[bin_idx]
        if cumulative_volume >= target_volume:
            break
    
    value_area_bins.sort()  # Sort bins by price level, not volume
    
    # Find Value Area High (VAH) and Value Area Low (VAL)
    vah = bin_edges[value_area_bins[-1] + 1]
    val = bin_edges[value_area_bins[0]]
    
    # Find High Volume Nodes (HVN) and Low Volume Nodes (LVN)
    median_volume = np.median(bin_volumes)
    hvn = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins) if bin_volumes[i] > median_volume * 1.5]
    lvn = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins) if bin_volumes[i] < median_volume * 0.5]
    
    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "hvn": hvn,
        "lvn": lvn
    }

def harmonic_patterns(high, low, close, tolerance=0.05):
    """
    Detect potential Harmonic patterns in price data
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    tolerance : float
        Tolerance for pattern ratio validation
        
    Returns:
    --------
    dict
        Dictionary containing detected patterns with their points and confidence
    """
    import pandas as pd
    import numpy as np
    
    # Function to check if a ratio is within tolerance
    def is_within_tolerance(actual, target, tol):
        return abs(actual - target) <= tol
    
    # Find potential swing points using zigzag
    def find_swing_points(price, depth=5):
        # Simple zigzag implementation
        swings = []
        swing_types = []  # 1 for high, -1 for low
        
        for i in range(depth, len(price) - depth):
            # Check if this is a local maximum
            if all(price.iloc[i] > price.iloc[i-j] for j in range(1, depth+1)) and \
               all(price.iloc[i] > price.iloc[i+j] for j in range(1, depth+1)):
                swings.append(i)
                swing_types.append(1)
            
            # Check if this is a local minimum
            elif all(price.iloc[i] < price.iloc[i-j] for j in range(1, depth+1)) and \
                 all(price.iloc[i] < price.iloc[i+j] for j in range(1, depth+1)):
                swings.append(i)
                swing_types.append(-1)
        
        return swings, swing_types
    
    # Get swing points
    typical_price = (high + low + close) / 3
    swings, types = find_swing_points(typical_price)
    
    # Need at least 5 swing points for harmonic patterns
    if len(swings) < 5:
        return {"patterns": []}
    
    patterns = []
    
    # Check the last few potential XABCD patterns
    for i in range(len(swings) - 4):
        # Get points
        x_idx, a_idx, b_idx, c_idx, d_idx = swings[i:i+5]
        
        # Skip if not alternating types
        if not (types[i] != types[i+1] and types[i+1] != types[i+2] and 
                types[i+2] != types[i+3] and types[i+3] != types[i+4]):
            continue
        
        # Get prices at swing points
        x = typical_price.iloc[x_idx]
        a = typical_price.iloc[a_idx]
        b = typical_price.iloc[b_idx]
        c = typical_price.iloc[c_idx]
        d = typical_price.iloc[d_idx]
        
        # Calculate ratios
        ab = abs((b - a) / (x - a))
        bc = abs((c - b) / (a - b))
        cd = abs((d - c) / (b - c))
        xd = abs((d - x) / (a - x))
        
        # Check for Gartley pattern
        gartley_confidence = 0
        
        # AB should be ~0.618
        if is_within_tolerance(ab, 0.618, tolerance):
            gartley_confidence += 25
        
        # BC should be ~0.382 or 0.886
        if is_within_tolerance(bc, 0.382, tolerance) or is_within_tolerance(bc, 0.886, tolerance):
            gartley_confidence += 25
        
        # CD should be ~1.272 or 1.618
        if is_within_tolerance(cd, 1.272, tolerance) or is_within_tolerance(cd, 1.618, tolerance):
            gartley_confidence += 25
        
        # XD should be ~0.786
        if is_within_tolerance(xd, 0.786, tolerance):
            gartley_confidence += 25
        
        if gartley_confidence > 50:
            patterns.append({
                "type": "Gartley",
                "confidence": gartley_confidence,
                "points": {
                    "X": x_idx,
                    "A": a_idx,
                    "B": b_idx,
                    "C": c_idx,
                    "D": d_idx
                },
                "completion_price": d
            })
        
        # Check for Butterfly pattern
        butterfly_confidence = 0
        
        # AB should be ~0.786
        if is_within_tolerance(ab, 0.786, tolerance):
            butterfly_confidence += 25
        
        # BC should be ~0.382 or 0.886
        if is_within_tolerance(bc, 0.382, tolerance) or is_within_tolerance(bc, 0.886, tolerance):
            butterfly_confidence += 25
        
        # CD should be ~1.618 or 2.618 or 2.0
        if (is_within_tolerance(cd, 1.618, tolerance) or 
            is_within_tolerance(cd, 2.618, tolerance) or 
            is_within_tolerance(cd, 2.0, tolerance)):
            butterfly_confidence += 25
        
        # XD should be ~1.27
        if is_within_tolerance(xd, 1.27, tolerance):
            butterfly_confidence += 25
        
        if butterfly_confidence > 50:
            patterns.append({
                "type": "Butterfly",
                "confidence": butterfly_confidence,
                "points": {
                    "X": x_idx,
                    "A": a_idx,
                    "B": b_idx,
                    "C": c_idx,
                    "D": d_idx
                },
                "completion_price": d
            })
        
        # Check for Bat pattern
        bat_confidence = 0
        
        # AB should be ~0.382 or 0.5
        if is_within_tolerance(ab, 0.382, tolerance) or is_within_tolerance(ab, 0.5, tolerance):
            bat_confidence += 25
        
        # BC should be ~0.382 or 0.886
        if is_within_tolerance(bc, 0.382, tolerance) or is_within_tolerance(bc, 0.886, tolerance):
            bat_confidence += 25
        
        # CD should be ~1.618 or 2.618
        if is_within_tolerance(cd, 1.618, tolerance) or is_within_tolerance(cd, 2.618, tolerance):
            bat_confidence += 25
        
        # XD should be ~0.886
        if is_within_tolerance(xd, 0.886, tolerance):
            bat_confidence += 25
        
        if bat_confidence > 50:
            patterns.append({
                "type": "Bat",
                "confidence": bat_confidence,
                "points": {
                    "X": x_idx,
                    "A": a_idx,
                    "B": b_idx,
                    "C": c_idx,
                    "D": d_idx
                },
                "completion_price": d
            })
    
    return {"patterns": patterns}

def divergence_scanner(price, oscillator, window=20):
    """
    Scan for regular and hidden divergences between price and an oscillator
    
    Parameters:
    -----------
    price : pd.Series
        Series of price data (typically close prices)
    oscillator : pd.Series
        Series of oscillator values (e.g. RSI, MACD, etc.)
    window : int
        Number of periods to look back for divergence
        
    Returns:
    --------
    dict
        Dictionary containing regular and hidden divergences with their strength
    """
    if len(price) < window or len(oscillator) < window:
        return {
            "regular_bullish": False,
            "regular_bearish": False,
            "hidden_bullish": False,
            "hidden_bearish": False,
            "strength": 0
        }
    
    # Get the window of data
    price_window = price[-window:].copy()
    osc_window = oscillator[-window:].copy()
    
    # Find local minima and maxima in price
    price_peaks = []
    price_troughs = []
    
    for i in range(2, len(price_window) - 2):
        # Check for local maximum
        if (price_window.iloc[i] > price_window.iloc[i-1] and 
            price_window.iloc[i] > price_window.iloc[i-2] and
            price_window.iloc[i] > price_window.iloc[i+1] and
            price_window.iloc[i] > price_window.iloc[i+2]):
            price_peaks.append(i)
        
        # Check for local minimum
        if (price_window.iloc[i] < price_window.iloc[i-1] and 
            price_window.iloc[i] < price_window.iloc[i-2] and 
            price_window.iloc[i] < price_window.iloc[i+1] and
            price_window.iloc[i] < price_window.iloc[i+2]):
            price_troughs.append(i)
    
    # Find local minima and maxima in oscillator
    osc_peaks = []
    osc_troughs = []
    
    for i in range(2, len(osc_window) - 2):
        # Check for local maximum
        if (osc_window.iloc[i] > osc_window.iloc[i-1] and 
            osc_window.iloc[i] > osc_window.iloc[i-2] and
            osc_window.iloc[i] > osc_window.iloc[i+1] and
            osc_window.iloc[i] > osc_window.iloc[i+2]):
            osc_peaks.append(i)
        
        # Check for local minimum
        if (osc_window.iloc[i] < osc_window.iloc[i-1] and 
            osc_window.iloc[i] < osc_window.iloc[i-2] and 
            osc_window.iloc[i] < osc_window.iloc[i+1] and
            osc_window.iloc[i] < osc_window.iloc[i+2]):
            osc_troughs.append(i)
    
    # Need at least 2 peaks and troughs to check for divergence
    if len(price_peaks) < 2 or len(price_troughs) < 2 or len(osc_peaks) < 2 or len(osc_troughs) < 2:
        return {
            "regular_bullish": False,
            "regular_bearish": False,
            "hidden_bullish": False,
            "hidden_bearish": False,
            "strength": 0
        }
    
    # Check for regular bullish divergence (lower lows in price, higher lows in oscillator)
    regular_bullish = False
    if price_window.iloc[price_troughs[-1]] < price_window.iloc[price_troughs[-2]] and \
       osc_window.iloc[osc_troughs[-1]] > osc_window.iloc[osc_troughs[-2]]:
        regular_bullish = True
    
    # Check for regular bearish divergence (higher highs in price, lower highs in oscillator)
    regular_bearish = False
    if price_window.iloc[price_peaks[-1]] > price_window.iloc[price_peaks[-2]] and \
       osc_window.iloc[osc_peaks[-1]] < osc_window.iloc[osc_peaks[-2]]:
        regular_bearish = True
    
    # Check for hidden bullish divergence (higher lows in price, lower lows in oscillator)
    hidden_bullish = False
    if price_window.iloc[price_troughs[-1]] > price_window.iloc[price_troughs[-2]] and \
       osc_window.iloc[osc_troughs[-1]] < osc_window.iloc[osc_troughs[-2]]:
        hidden_bullish = True
    
    # Check for hidden bearish divergence (lower highs in price, higher highs in oscillator)
    hidden_bearish = False
    if price_window.iloc[price_peaks[-1]] < price_window.iloc[price_peaks[-2]] and \
       osc_window.iloc[osc_peaks[-1]] > osc_window.iloc[osc_peaks[-2]]:
        hidden_bearish = True
    
    # Calculate divergence strength (percentage difference)
    strength = 0
    if regular_bullish or hidden_bullish:
        price_change = abs((price_window.iloc[price_troughs[-1]] / price_window.iloc[price_troughs[-2]]) - 1) * 100
        osc_change = abs((osc_window.iloc[osc_troughs[-1]] / osc_window.iloc[osc_troughs[-2]]) - 1) * 100
        strength = (price_change + osc_change) / 2
    elif regular_bearish or hidden_bearish:
        price_change = abs((price_window.iloc[price_peaks[-1]] / price_window.iloc[price_peaks[-2]]) - 1) * 100
        osc_change = abs((osc_window.iloc[osc_peaks[-1]] / osc_window.iloc[osc_peaks[-2]]) - 1) * 100
        strength = (price_change + osc_change) / 2
    
    return {
        "regular_bullish": regular_bullish,
        "regular_bearish": regular_bearish,
        "hidden_bullish": hidden_bullish,
        "hidden_bearish": hidden_bearish,
        "strength": strength
    }

def stochastic_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Calculate Stochastic RSI
    
    Parameters:
    -----------
    close : pd.Series
        Series of close prices
    rsi_period : int
        Period for RSI calculation
    stoch_period : int
        Period for Stochastic calculation
    k_period : int
        Smoothing for %K line
    d_period : int
        Smoothing for %D line
        
    Returns:
    --------
    pd.Series, pd.Series
        %K and %D values of Stochastic RSI
    """
    from app.technical_indicators.basic_indicators import relative_strength_index
    
    # Calculate RSI
    rsi = relative_strength_index(close, window=rsi_period)
    
    # Calculate Stochastic RSI
    stoch_rsi = pd.Series(index=close.index, dtype='float64')
    
    for i in range(stoch_period, len(rsi)):
        rsi_lookback = rsi[i-stoch_period+1:i+1]
        rsi_min = rsi_lookback.min()
        rsi_max = rsi_lookback.max()
        
        if rsi_max - rsi_min != 0:
            stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min)
        else:
            stoch_rsi[i] = 0
    
    # Calculate %K and %D
    k = 100 * stoch_rsi.rolling(window=k_period).mean()
    d = k.rolling(window=d_period).mean()
    
    return k, d

def elliott_wave_tracker(high, low, close, volume, threshold=0.3, window_size=50):
    """
    Track potential Elliott Wave patterns
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of close prices
    volume : pd.Series
        Series of volume data
    threshold : float
        Threshold for wave identification
    window_size : int
        Lookback window for pattern identification
        
    Returns:
    --------
    dict
        Dictionary containing potential Elliott Wave patterns and characteristics
    """
    # Identify potential swing points
    swings = []
    swing_types = []  # 1 for high, -1 for low
    
    # Simple swing identification
    for i in range(5, len(close) - 5):
        # Check if this is a local maximum
        if all(high.iloc[i] > high.iloc[i-j] for j in range(1, 5)) and \
           all(high.iloc[i] > high.iloc[i+j] for j in range(1, 5)):
            swings.append(i)
            swing_types.append(1)
        
        # Check if this is a local minimum
        elif all(low.iloc[i] < low.iloc[i-j] for j in range(1, 5)) and \
             all(low.iloc[i] < low.iloc[i+j] for j in range(1, 5)):
            swings.append(i)
            swing_types.append(-1)
    
    # Not enough swing points to identify patterns
    if len(swings) < 9:
        return {
            "wave_count": 0,
            "impulse_waves": [],
            "corrective_waves": [],
            "current_position": "undefined",
            "confidence": 0.0
        }
    
    # Analyze the last potential pattern within the window
    recent_swings = [i for i in swings if i >= len(close) - window_size]
    recent_types = [swing_types[swings.index(i)] for i in recent_swings]
    
    # Need at least 9 points for a complete cycle (5 impulse + 3 corrective + final)
    if len(recent_swings) < 9:
        return {
            "wave_count": len(recent_swings),
            "impulse_waves": [],
            "corrective_waves": [],
            "current_position": "developing",
            "confidence": 0.0
        }
    
    # Get prices at swing points
    swing_prices = [close.iloc[i] for i in recent_swings]
    
    # Check for impulse wave pattern
    impulse_waves = []
    corrective_waves = []
    confidence = 0.0
    
    # Basic check for 5-3 pattern
    # Impulse waves should be in the trend direction
    # Waves 1, 3, 5 should be in primary trend direction
    # Waves 2, 4 should be corrections
    
    # Find 5 alternating waves first (potential impulse)
    i = 0
    while i < len(recent_types) - 4:
        if (recent_types[i] == recent_types[i+2] == recent_types[i+4] and
            recent_types[i+1] == recent_types[i+3] and
            recent_types[i] != recent_types[i+1]):
            
            impulse_start = recent_swings[i]
            impulse_waves = [
                {"wave": 1, "index": recent_swings[i], "price": swing_prices[i]},
                {"wave": 2, "index": recent_swings[i+1], "price": swing_prices[i+1]},
                {"wave": 3, "index": recent_swings[i+2], "price": swing_prices[i+2]},
                {"wave": 4, "index": recent_swings[i+3], "price": swing_prices[i+3]},
                {"wave": 5, "index": recent_swings[i+4], "price": swing_prices[i+4]}
            ]
            
            # Check for corrective waves
            if i + 7 < len(recent_swings):
                corrective_start = recent_swings[i+5]
                corrective_waves = [
                    {"wave": "A", "index": recent_swings[i+5], "price": swing_prices[i+5]},
                    {"wave": "B", "index": recent_swings[i+6], "price": swing_prices[i+6]},
                    {"wave": "C", "index": recent_swings[i+7], "price": swing_prices[i+7]}
                ]
                
                # Check if the pattern follows Elliott Wave rules
                # 1. Wave 3 should not be the shortest among 1, 3, 5
                # 2. Wave 4 should not overlap with Wave 1
                if ((abs(impulse_waves[2]["price"] - impulse_waves[0]["price"]) > 
                    abs(impulse_waves[0]["price"] - impulse_waves[0]["price"]) or
                    abs(impulse_waves[2]["price"] - impulse_waves[0]["price"]) > 
                    abs(impulse_waves[4]["price"] - impulse_waves[3]["price"])) and
                    ((recent_types[i] == 1 and impulse_waves[3]["price"] > impulse_waves[0]["price"]) or
                     (recent_types[i] == -1 and impulse_waves[3]["price"] < impulse_waves[0]["price"]))):
                    
                    confidence = 0.7  # Found pattern with reasonable confidence
                    break
            
            # If we found impulse but not enough points for corrective
            elif i + 5 < len(recent_swings):
                confidence = 0.4  # Only identified impulse waves
                break
                
        i += 1
    
    # Determine current position
    if confidence > 0:
        if impulse_waves and len(corrective_waves) == 3:
            current_position = "Corrective phase complete"
        elif impulse_waves and len(corrective_waves) < 3 and len(corrective_waves) > 0:
            current_position = f"Corrective wave {['A','B','C'][len(corrective_waves)-1]}"
        elif impulse_waves:
            current_position = f"Impulse wave {5 if len(impulse_waves) >= 5 else len(impulse_waves)} complete"
        else:
            current_position = "undefined"
    else:
        current_position = "No clear Elliott Wave pattern"
    
    return {
        "wave_count": len(impulse_waves) + len(corrective_waves),
        "impulse_waves": impulse_waves,
        "corrective_waves": corrective_waves,
        "current_position": current_position,
        "confidence": confidence
    }

def mean_reversion_index(close, high, low, period=14, std_dev_multiplier=2.0):
    """
    Calculate Mean Reversion Index to identify potential mean reversion opportunities
    
    Parameters:
    -----------
    close : pd.Series
        Series of close prices
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    period : int
        Lookback period
    std_dev_multiplier : float
        Standard deviation multiplier for bands
        
    Returns:
    --------
    pd.Series, pd.Series, pd.Series
        Mean Reversion Index, Upper Band, Lower Band
    """
    # Calculate simple moving average
    sma = close.rolling(window=period).mean()
    
    # Calculate standard deviation
    std_dev = close.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = sma + (std_dev * std_dev_multiplier)
    lower_band = sma - (std_dev * std_dev_multiplier)
    
    # Calculate distance from mean (normalized by standard deviation)
    mean_distance = (close - sma) / std_dev
    
    # Calculate mean reversion index (-1 to 1)
    mri = mean_distance.rolling(window=period).mean() * -1  # Invert so positive values suggest reversion to mean
    
    # Calculate true range and ATR for normalization
    true_range = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    atr = true_range.rolling(window=period).mean()
    
    # Normalize MRI based on historical range
    mri_min = mri.rolling(window=period*2).min()
    mri_max = mri.rolling(window=period*2).max()
    mri_range = mri_max - mri_min
    
    # Prevent division by zero
    mri_range = mri_range.replace(0, 1)
    
    normalized_mri = (mri - mri_min) / mri_range
    
    # Scale to -100 to 100 range
    scaled_mri = (normalized_mri * 200) - 100
    
    return scaled_mri, upper_band, lower_band

def market_breadth_indicators(advances, declines, unchanged, period=10):
    """
    Calculate Market Breadth Indicators for broader market analysis
    
    Parameters:
    -----------
    advances : pd.Series
        Series of number of advancing issues
    declines : pd.Series
        Series of number of declining issues
    unchanged : pd.Series
        Series of number of unchanged issues
    period : int
        Lookback period for calculations
        
    Returns:
    --------
    dict
        Dictionary containing various market breadth indicators
    """
    # Calculate Advance-Decline Line
    adl = (advances - declines).cumsum()
    
    # Calculate Advance-Decline Ratio
    adr = advances / declines
    
    # Calculate McClellan Oscillator
    ema19 = (advances - declines).ewm(span=19, adjust=False).mean()
    ema39 = (advances - declines).ewm(span=39, adjust=False).mean()
    mcclellan_oscillator = ema19 - ema39
    
    # Calculate McClellan Summation Index
    summation_index = mcclellan_oscillator.cumsum()
    
    # Calculate Arms Index (TRIN)
    trin = (advances / declines) / (advances.sum() / declines.sum())
    
    # Calculate Absolute Breadth Index
    abi = abs(advances - declines) / (advances + declines + unchanged)
    
    # Calculate High-Low Index
    high_low_index = advances / (advances + declines) * 100
    high_low_diff = advances - declines
    
    # Calculate Bullish Percent Index (using a simple rolling window approximation)
    bpi = (advances / (advances + declines) * 100).rolling(window=period).mean()
    
    return {
        "advance_decline_line": adl,
        "advance_decline_ratio": adr,
        "mcclellan_oscillator": mcclellan_oscillator,
        "mcclellan_summation_index": summation_index,
        "trin": trin,
        "absolute_breadth_index": abi,
        "high_low_index": high_low_index,
        "high_low_diff": high_low_diff,
        "bullish_percent_index": bpi
    }

def orderflow_analysis(price, volume, bid_volume, ask_volume, delta_threshold=0.5):
    """
    Perform basic order flow analysis to identify buying/selling pressure
    
    Parameters:
    -----------
    price : pd.Series
        Series of price data
    volume : pd.Series
        Series of total volume
    bid_volume : pd.Series
        Series of bid volume (buying volume)
    ask_volume : pd.Series
        Series of ask volume (selling volume)
    delta_threshold : float
        Threshold for significant delta imbalance (0-1)
        
    Returns:
    --------
    dict
        Dictionary containing order flow metrics
    """
    # Calculate basic delta (difference between buying and selling volume)
    delta = bid_volume - ask_volume
    
    # Calculate cumulative delta
    cumulative_delta = delta.cumsum()
    
    # Calculate delta percentage (normalized by total volume)
    delta_percent = delta / volume
    
    # Identify significant imbalances
    significant_buying = delta_percent > delta_threshold
    significant_selling = delta_percent < -delta_threshold
    
    # Calculate imbalance ratio
    imbalance_ratio = bid_volume / ask_volume
    
    # Calculate volume weighted average price (VWAP)
    vwap = (price * volume).cumsum() / volume.cumsum()
    
    # Calculate Volume at Price Points (simplified)
    price_rounded = price.round(decimals=2)  # Round to 2 decimal places for grouping
    volume_profile = volume.groupby(price_rounded).sum()
    
    # Calculate average trade size
    average_trade_size = volume / volume.count()
    
    # Identify potential absorption zones (high volume with minimal price movement)
    price_change = price.pct_change()
    volume_to_price_ratio = volume / (abs(price_change) + 0.0001)  # Add small value to prevent division by zero
    absorption_zones = volume_to_price_ratio > volume_to_price_ratio.rolling(window=20).mean() * 2
    
    return {
        "delta": delta,
        "cumulative_delta": cumulative_delta,
        "delta_percent": delta_percent,
        "significant_buying": significant_buying,
        "significant_selling": significant_selling,
        "imbalance_ratio": imbalance_ratio,
        "vwap": vwap,
        "volume_profile": volume_profile,
        "average_trade_size": average_trade_size,
        "absorption_zones": absorption_zones
    } 