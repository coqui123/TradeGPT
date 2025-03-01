"""
Smart Money Indicators
Contains advanced indicators focused on institutional behavior and liquidity
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


def liquidity_sweep_analysis(high: pd.Series, low: pd.Series, close: pd.Series, 
                            volume: pd.Series, lookback: int = 10, 
                            threshold: float = 1.5) -> Dict[str, Any]:
    """
    Identify potential liquidity sweeps where price briefly breaks a level and reverses
    
    Parameters:
    -----------
    high, low, close: pd.Series
        Price data
    volume: pd.Series
        Volume data
    lookback: int
        Period to look back for swing highs/lows
    threshold: float
        Threshold for determining significant volume spike
    
    Returns:
    --------
    dict: Dictionary containing identified sweep zones and their strengths
    """
    sweeps = {
        'high_sweeps': [],
        'low_sweeps': [],
        'current_high_sweep': False,
        'current_low_sweep': False
    }
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(close)-lookback):
        # Check for swing high
        if high[i] == max(high[i-lookback:i+lookback+1]):
            swing_highs.append((i, high[i]))
        
        # Check for swing low
        if low[i] == min(low[i-lookback:i+lookback+1]):
            swing_lows.append((i, low[i]))
    
    # Check for sweeps above swing highs
    for idx, level in swing_highs[:-1]:
        for j in range(idx+1, min(idx+20, len(close)-1)):
            if high[j] > level and close[j] < level and volume[j] > volume[j-1]*threshold:
                sweeps['high_sweeps'].append({
                    'index': j,
                    'price': float(level),
                    'strength': float((volume[j]/volume[j-5:j].mean())),
                    'subsequent_move': float(level - min(low[j:min(j+5, len(close)-1)]))
                })
    
    # Check for sweeps below swing lows
    for idx, level in swing_lows[:-1]:
        for j in range(idx+1, min(idx+20, len(close)-1)):
            if low[j] < level and close[j] > level and volume[j] > volume[j-1]*threshold:
                sweeps['low_sweeps'].append({
                    'index': j,
                    'price': float(level),
                    'strength': float((volume[j]/volume[j-5:j].mean())),
                    'subsequent_move': float(max(high[j:min(j+5, len(close)-1)]) - level)
                })
    
    # Check for current potential sweeps
    if len(swing_highs) > 0 and high.iloc[-1] > swing_highs[-1][1] and close.iloc[-1] < swing_highs[-1][1]:
        sweeps['current_high_sweep'] = True
    
    if len(swing_lows) > 0 and low.iloc[-1] < swing_lows[-1][1] and close.iloc[-1] > swing_lows[-1][1]:
        sweeps['current_low_sweep'] = True
    
    return sweeps


def order_block_detection(open_price: pd.Series, high: pd.Series, low: pd.Series, 
                         close: pd.Series, volume: pd.Series, 
                         window: int = 14) -> Dict[str, List[Dict[str, Any]]]:
    """
    Identify potential order blocks based on price action and volume
    
    Parameters:
    -----------
    open_price, high, low, close: pd.Series
        Price data
    volume: pd.Series
        Volume data
    window: int
        Lookback period
    
    Returns:
    --------
    dict: Dictionary with bullish and bearish order blocks
    """
    bullish_blocks = []
    bearish_blocks = []
    
    # Calculate body size and total candle size
    body_size = abs(close - open_price)
    candle_size = high - low
    body_percent = body_size / candle_size
    
    # Calculate average metrics
    avg_volume = volume.rolling(window=window).mean()
    avg_candle_size = candle_size.rolling(window=window).mean()
    
    # Identify potential order blocks
    for i in range(window, len(close)-1):
        # Bullish order block (strong down candle followed by reversal)
        if (close.iloc[i] < open_price.iloc[i] and  # Down candle
            body_size.iloc[i] > 0.7 * candle_size.iloc[i] and  # Strong body
            volume.iloc[i] > 1.5 * avg_volume.iloc[i] and  # High volume
            close.iloc[i+1] > open_price.iloc[i+1] and  # Next candle is up
            high.iloc[i+1] > high.iloc[i]):  # Breaking resistance
            
            bullish_blocks.append({
                'index': i,
                'low': float(low.iloc[i]),
                'high': float(open_price.iloc[i]),
                'strength': float(volume.iloc[i] / avg_volume.iloc[i]),
                'candle_size_ratio': float(candle_size.iloc[i] / avg_candle_size.iloc[i])
            })
        
        # Bearish order block (strong up candle followed by reversal)
        if (close.iloc[i] > open_price.iloc[i] and  # Up candle
            body_size.iloc[i] > 0.7 * candle_size.iloc[i] and  # Strong body
            volume.iloc[i] > 1.5 * avg_volume.iloc[i] and  # High volume
            close.iloc[i+1] < open_price.iloc[i+1] and  # Next candle is down
            low.iloc[i+1] < low.iloc[i]):  # Breaking support
            
            bearish_blocks.append({
                'index': i,
                'low': float(close.iloc[i]),
                'high': float(open_price.iloc[i]),
                'strength': float(volume.iloc[i] / avg_volume.iloc[i]),
                'candle_size_ratio': float(candle_size.iloc[i] / avg_candle_size.iloc[i])
            })
    
    # Find active order blocks (recent ones that haven't been broken)
    active_bullish = []
    for block in bullish_blocks:
        idx = block['index']
        if min(low.iloc[idx+1:]) > block['low']:  # Support hasn't been broken
            active_bullish.append(block)
    
    active_bearish = []
    for block in bearish_blocks:
        idx = block['index']
        if max(high.iloc[idx+1:]) < block['high']:  # Resistance hasn't been broken
            active_bearish.append(block)
    
    return {
        'bullish_order_blocks': bullish_blocks,
        'bearish_order_blocks': bearish_blocks,
        'active_bullish_blocks': active_bullish,
        'active_bearish_blocks': active_bearish
    }


def smart_money_analysis(open_price: pd.Series, high: pd.Series, low: pd.Series, 
                        close: pd.Series, volume: pd.Series) -> Dict[str, Any]:
    """
    Analyze market structure from a smart money perspective
    
    Parameters:
    -----------
    open_price, high, low, close, volume: pd.Series
        Price and volume data
    
    Returns:
    --------
    dict: Smart money analysis results
    """
    # Identify Fair Value Gaps (FVG)
    bullish_fvg = []
    bearish_fvg = []
    
    for i in range(2, len(close)):
        # Bullish FVG (gap up)
        if low.iloc[i] > high.iloc[i-2]:
            bullish_fvg.append({
                'index': i,
                'top': float(low.iloc[i]),
                'bottom': float(high.iloc[i-2]),
                'size': float(low.iloc[i] - high.iloc[i-2]),
                'mitigated': bool(min(low.iloc[i:]) <= high.iloc[i-2])
            })
        
        # Bearish FVG (gap down)
        if high.iloc[i] < low.iloc[i-2]:
            bearish_fvg.append({
                'index': i,
                'top': float(low.iloc[i-2]),
                'bottom': float(high.iloc[i]),
                'size': float(low.iloc[i-2] - high.iloc[i]),
                'mitigated': bool(max(high.iloc[i:]) >= low.iloc[i-2])
            })
    
    # Identify Equal Highs/Lows (Double tops/bottoms)
    equal_highs = []
    equal_lows = []
    threshold = 0.0005  # 0.05% threshold for equality
    
    for i in range(10, len(close)):
        for j in range(max(0, i-20), i-5):
            # Equal highs
            if abs(high.iloc[i] - high.iloc[j])/high.iloc[j] < threshold:
                equal_highs.append({
                    'index1': j,
                    'index2': i,
                    'price': float(high.iloc[i]),
                    'swept': bool(any(h > high.iloc[i] for h in high.iloc[i+1:min(i+10, len(high))]))
                })
            
            # Equal lows
            if abs(low.iloc[i] - low.iloc[j])/low.iloc[j] < threshold:
                equal_lows.append({
                    'index1': j,
                    'index2': i,
                    'price': float(low.iloc[i]),
                    'swept': bool(any(l < low.iloc[i] for l in low.iloc[i+1:min(i+10, len(low))]))
                })
    
    # Identify Breaker Blocks (broken S/R that becomes opposite)
    breaker_blocks = []
    
    # Combine results
    return {
        'bullish_fvg': bullish_fvg,
        'bearish_fvg': bearish_fvg,
        'equal_highs': equal_highs,
        'equal_lows': equal_lows,
        'breaker_blocks': breaker_blocks
    }


def cumulative_delta_analysis(open_price: pd.Series, high: pd.Series, low: pd.Series, 
                             close: pd.Series, volume: pd.Series, 
                             bid_volume: Optional[pd.Series] = None, 
                             ask_volume: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Analyze cumulative delta (buying vs selling pressure)
    
    Parameters:
    -----------
    open_price, high, low, close, volume: pd.Series
        Price and volume data
    bid_volume, ask_volume: pd.Series, optional
        If available, the actual bid/ask volume data
    
    Returns:
    --------
    dict: Cumulative delta analysis results
    """
    # If we don't have bid/ask volume, estimate it
    if bid_volume is None or ask_volume is None:
        # Estimate based on price action
        delta = pd.Series(index=close.index)
        for i in range(len(close)):
            if close.iloc[i] > open_price.iloc[i]:
                # Bullish candle - estimate more buying
                delta.iloc[i] = volume.iloc[i] * (close.iloc[i] - open_price.iloc[i]) / (high.iloc[i] - low.iloc[i])
            else:
                # Bearish candle - estimate more selling
                delta.iloc[i] = -volume.iloc[i] * (open_price.iloc[i] - close.iloc[i]) / (high.iloc[i] - low.iloc[i])
    else:
        # We have actual bid/ask data
        delta = ask_volume - bid_volume
    
    # Calculate cumulative delta
    cum_delta = delta.cumsum()
    
    # Calculate delta divergence with price
    delta_div = []
    for i in range(20, len(close)):
        if close.iloc[i] > close.iloc[i-10] and cum_delta.iloc[i] < cum_delta.iloc[i-10]:
            delta_div.append({
                'index': i,
                'type': 'bearish',
                'price': float(close.iloc[i]),
                'delta': float(cum_delta.iloc[i])
            })
        elif close.iloc[i] < close.iloc[i-10] and cum_delta.iloc[i] > cum_delta.iloc[i-10]:
            delta_div.append({
                'index': i,
                'type': 'bullish',
                'price': float(close.iloc[i]),
                'delta': float(cum_delta.iloc[i])
            })
    
    # Calculate delta momentum
    delta_momentum = delta.rolling(window=14).mean()
    
    # Calculate delta percent (relative to volume)
    delta_percent = (delta / volume) * 100
    
    # Calculate imbalance ratio
    imbalance_ratio = abs(delta) / volume
    
    return {
        'delta': delta.fillna(0).tolist(),
        'cumulative_delta': cum_delta.fillna(0).tolist(),
        'delta_divergences': delta_div,
        'delta_momentum': delta_momentum.fillna(0).tolist(),
        'delta_percent': delta_percent.fillna(0).tolist(),
        'imbalance_ratio': imbalance_ratio.fillna(0).tolist()
    }


def market_depth_analysis(price: float, bid_levels: List[float], 
                         ask_levels: List[float], volumes: List[float]) -> Dict[str, Any]:
    """
    Analyze market depth to identify liquidity zones
    
    Parameters:
    -----------
    price: float
        Current price
    bid_levels: list
        List of bid price levels
    ask_levels: list
        List of ask price levels
    volumes: list
        Corresponding volumes at each level
    
    Returns:
    --------
    dict: Market depth analysis
    """
    # Calculate bid/ask imbalance
    total_bid_volume = sum(vol for p, vol in zip(bid_levels, volumes) if p < price)
    total_ask_volume = sum(vol for p, vol in zip(ask_levels, volumes) if p > price)
    
    imbalance_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
    
    # Find liquidity clusters
    bid_clusters = []
    ask_clusters = []
    
    # Identify large bid walls (support)
    for i in range(len(bid_levels)):
        if i > 0 and bid_levels[i] < bid_levels[i-1]:  # Ensure sorted
            relative_size = volumes[i] / sum(volumes[:i])
            if relative_size > 0.1:  # Large relative size
                bid_clusters.append({
                    'price': float(bid_levels[i]),
                    'volume': float(volumes[i]),
                    'relative_size': float(relative_size)
                })
    
    # Identify large ask walls (resistance)
    for i in range(len(ask_levels)):
        if i > 0 and ask_levels[i] > ask_levels[i-1]:  # Ensure sorted
            relative_size = volumes[i] / sum(volumes[:i])
            if relative_size > 0.1:  # Large relative size
                ask_clusters.append({
                    'price': float(ask_levels[i]),
                    'volume': float(volumes[i]),
                    'relative_size': float(relative_size)
                })
    
    return {
        'bid_ask_imbalance': float(imbalance_ratio),
        'buy_pressure': float(total_bid_volume),
        'sell_pressure': float(total_ask_volume),
        'support_clusters': bid_clusters,
        'resistance_clusters': ask_clusters
    }


def volatility_regime_detection(close: pd.Series, high: pd.Series, low: pd.Series, 
                              lookback: int = 100, short_window: int = 5, 
                              long_window: int = 20) -> Dict[str, Any]:
    """
    Detect volatility regime and adapt indicators accordingly
    
    Parameters:
    -----------
    close, high, low: pd.Series
        Price data
    lookback: int
        Historical lookback for establishing baseline
    short_window, long_window: int
        Short and long windows for volatility calculation
    
    Returns:
    --------
    dict: Volatility regime information
    """
    # Calculate short and long-term ATR
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    short_atr = tr.rolling(window=short_window).mean()
    long_atr = tr.rolling(window=long_window).mean()
    
    # Calculate volatility ratio
    volatility_ratio = short_atr / long_atr
    
    # Determine historical volatility percentiles
    hist_vol = tr.rolling(window=20).std()
    percentiles = [10, 25, 50, 75, 90]
    vol_percentiles = {p: np.nanpercentile(hist_vol.iloc[:min(lookback, len(hist_vol))], p) for p in percentiles}
    
    # Current volatility percentile
    current_vol = hist_vol.iloc[-1]
    current_percentile = 0
    for p in sorted(percentiles):
        if current_vol >= vol_percentiles[p]:
            current_percentile = p
    
    # Define the regime
    regime = "normal_volatility"
    if not pd.isna(volatility_ratio.iloc[-1]):
        if volatility_ratio.iloc[-1] > 1.5:
            regime = "expanding_volatility"
        elif volatility_ratio.iloc[-1] < 0.75:
            regime = "contracting_volatility"
    
    # Adaptive indicator settings based on regime
    indicator_adjustments = {
        "expanding_volatility": {
            "rsi_period": 10,  # Shorter for faster response
            "macd_fast": 8,    # Faster settings
            "macd_slow": 17,
            "atr_multiplier": 2.5,  # Wider stops
            "bollinger_std": 2.5     # Wider bands
        },
        "contracting_volatility": {
            "rsi_period": 21,  # Longer for less noise
            "macd_fast": 12,   # Standard settings
            "macd_slow": 26,
            "atr_multiplier": 1.5,  # Tighter stops
            "bollinger_std": 1.8     # Narrower bands
        },
        "normal_volatility": {
            "rsi_period": 14,  # Standard settings
            "macd_fast": 12,
            "macd_slow": 26,
            "atr_multiplier": 2.0,
            "bollinger_std": 2.0
        }
    }
    
    return {
        'regime': regime,
        'volatility_percentile': float(current_percentile),
        'volatility_ratio': float(volatility_ratio.iloc[-1]) if not pd.isna(volatility_ratio.iloc[-1]) else 1.0,
        'indicator_adjustments': indicator_adjustments[regime],
        'short_atr': float(short_atr.iloc[-1]) if not pd.isna(short_atr.iloc[-1]) else 0.0,
        'long_atr': float(long_atr.iloc[-1]) if not pd.isna(long_atr.iloc[-1]) else 0.0
    }


def funding_liquidation_analysis(price: float, funding_rate: pd.Series, 
                                long_liquidations: pd.Series, 
                                short_liquidations: pd.Series, 
                                open_interest: pd.Series) -> Dict[str, Any]:
    """
    Analyze funding rates and liquidation levels for crypto futures
    
    Parameters:
    -----------
    price: float
        Current price
    funding_rate: pd.Series
        Historical funding rates
    long_liquidations, short_liquidations: pd.Series
        Historical liquidation data
    open_interest: pd.Series
        Open interest data
    
    Returns:
    --------
    dict: Funding and liquidation analysis
    """
    # Calculate average funding rate
    avg_funding = funding_rate.rolling(window=8).mean()  # 8 funding periods = 1 day
    
    # Identify extreme funding (market sentiment)
    extreme_positive = funding_rate > 0.001  # 0.1% is considered high
    extreme_negative = funding_rate < -0.001  # -0.1% is considered low
    
    # Calculate funding rate trend
    funding_trend = "neutral"
    if len(avg_funding) >= 3:
        if avg_funding.iloc[-3:].mean() > 0.0005:
            funding_trend = "bullish"  # High positive funding = bullish sentiment
        elif avg_funding.iloc[-3:].mean() < -0.0005:
            funding_trend = "bearish"  # High negative funding = bearish sentiment
    
    # Calculate liquidation clusters
    long_liq_clusters = []
    short_liq_clusters = []
    
    # Process long liquidations to find clusters
    if not long_liquidations.empty:
        for i in range(1, len(long_liquidations)):
            if long_liquidations.iloc[i] > long_liquidations.iloc[:i].mean() * 3:  # 3x average
                long_liq_clusters.append({
                    'price': float(price),  # Using current price as a proxy
                    'volume': float(long_liquidations.iloc[i]),
                    'date': str(long_liquidations.index[i])
                })
    
    # Process short liquidations to find clusters
    if not short_liquidations.empty:
        for i in range(1, len(short_liquidations)):
            if short_liquidations.iloc[i] > short_liquidations.iloc[:i].mean() * 3:
                short_liq_clusters.append({
                    'price': float(price),  # Using current price as a proxy
                    'volume': float(short_liquidations.iloc[i]),
                    'date': str(short_liquidations.index[i])
                })
    
    # Analyze open interest
    oi_change = 0.0
    if len(open_interest) >= 3:
        oi_change = open_interest.pct_change(3).iloc[-1] * 100  # 3-period change in percent
    
    market_sentiment = "neutral"
    if oi_change > 10 and funding_trend == "bullish":
        market_sentiment = "overwhelmingly_bullish"
    elif oi_change > 10 and funding_trend == "bearish":
        market_sentiment = "overwhelmingly_bearish"
    elif oi_change > 0 and funding_trend == "bullish":
        market_sentiment = "bullish"
    elif oi_change < 0 and funding_trend == "bearish":
        market_sentiment = "bearish"
    
    return {
        'funding_trend': funding_trend,
        'current_funding': float(funding_rate.iloc[-1]) if not funding_rate.empty else 0.0,
        'avg_funding': float(avg_funding.iloc[-1]) if not avg_funding.empty else 0.0,
        'funding_extremes': {
            'positive_count': int(extreme_positive.sum()),
            'negative_count': int(extreme_negative.sum())
        },
        'long_liquidation_clusters': long_liq_clusters,
        'short_liquidation_clusters': short_liq_clusters,
        'open_interest_change': float(oi_change),
        'market_sentiment': market_sentiment
    }


def cross_asset_correlation(main_asset: pd.Series, 
                           related_assets: Dict[str, pd.Series], 
                           lookback_periods: List[int] = [14, 30, 90]) -> Dict[str, Any]:
    """
    Analyze correlations between the main asset and related markets
    
    Parameters:
    -----------
    main_asset: pd.Series
        Price data for the main asset
    related_assets: dict
        Dictionary of related asset price series
    lookback_periods: list
        Periods to calculate correlation over
    
    Returns:
    --------
    dict: Correlation analysis results
    """
    results = {}
    
    for period in lookback_periods:
        period_results = {}
        
        for name, asset_price in related_assets.items():
            # Ensure same length
            min_len = min(len(main_asset), len(asset_price))
            
            if min_len <= period:
                continue
                
            # Get returns
            main_returns = main_asset.pct_change().iloc[-period:min_len]
            asset_returns = asset_price.pct_change().iloc[-period:min_len]
            
            # Calculate correlation
            correlation = main_returns.corr(asset_returns)
            
            period_results[name] = {
                'correlation': float(correlation),
                'strength': 'strong_positive' if correlation > 0.7 else
                           'moderate_positive' if correlation > 0.3 else
                           'weak_positive' if correlation > 0 else
                           'weak_negative' if correlation > -0.3 else
                           'moderate_negative' if correlation > -0.7 else
                           'strong_negative'
            }
        
        results[f'{period}d'] = period_results
    
    # Find strongest correlations
    strongest_positive = ('none', 0)
    strongest_negative = ('none', 0)
    
    if len(results) > 0 and f'{lookback_periods[0]}d' in results:
        for asset, data in results[f'{lookback_periods[0]}d'].items():
            corr = data['correlation']
            if corr > strongest_positive[1]:
                strongest_positive = (asset, corr)
            if corr < strongest_negative[1]:
                strongest_negative = (asset, corr)
    
    return {
        'correlations': results,
        'strongest_positive': {
            'asset': strongest_positive[0],
            'correlation': float(strongest_positive[1])
        },
        'strongest_negative': {
            'asset': strongest_negative[0],
            'correlation': float(strongest_negative[1])
        }
    } 