def summarize_pair_data(pair_data):
    """
    Create a concise summary of pair data for LLM analysis
    """
    # ... existing code ...

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
        formatted["smart_money_concepts"]["liquidity_sweeps"] = indicators["liquidity_sweeps"]
        if indicators["liquidity_sweeps"].get("current_high_sweep") or indicators["liquidity_sweeps"].get("current_low_sweep"):
            formatted["overview"]["institutional_activity"]["detected"] = True
            formatted["overview"]["institutional_activity"]["description"] = "Recent liquidity sweep detected"
    
    if "order_blocks" in indicators:
        formatted["smart_money_concepts"]["order_blocks"] = indicators["order_blocks"]
        if indicators["order_blocks"].get("active_bullish_count", 0) > 0 or indicators["order_blocks"].get("active_bearish_count", 0) > 0:
            formatted["overview"]["institutional_activity"]["detected"] = True
            formatted["overview"]["institutional_activity"]["description"] = "Active order blocks detected"
    
    if "fair_value_gaps" in indicators:
        formatted["smart_money_concepts"]["fair_value_gaps"] = indicators["fair_value_gaps"]
    
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
                    ma_trends[ma_name] = value
        
        # Process exponential moving averages  
        if "exponential" in moving_averages:
            exp_mas = moving_averages["exponential"]
            for period in ["20", "50", "100", "200"]:
                if period in exp_mas:
                    ma_name = f"EMA_{period}"
                    value = exp_mas[period].get("value")
                    ma_trends[ma_name] = value
        
        # Add moving average trends
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
                formatted["volatility_indicators"]["bollinger_bands"] = {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "width": (upper - lower) / middle if middle != 0 else 0,
                    "price_position": "upper_band" if indicators.get("close", [])[-1] >= upper else 
                                      "lower_band" if indicators.get("close", [])[-1] <= lower else "middle_range"
                }
    
    # Process volume indicators
    volume_indicators = {}
    for indicator in ["OBV", "CMF", "MFI", "Chaikin_Oscillator", "VWAP"]:
        if indicator in indicators and isinstance(indicators[indicator], dict) and "value" in indicators[indicator]:
            volume_indicators[indicator] = indicators[indicator]["value"]
    
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