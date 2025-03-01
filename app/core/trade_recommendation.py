"""
Trade Recommendation Engine
Contains functions for generating trade recommendations
"""
import logging
import json
from typing import Dict, Any, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor

from app.core.analysis_engine import (
    calculate_technical_indicators, 
    summarize_pair_data,
    format_indicators_for_llm,
    add_smart_money_context
)
from app.core.llm_manager import LLMManager
from app.utils.json_utils import format_json_result
from langchain.schema import SystemMessage, UserMessage
from log import logger

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize LLM manager
llm_manager = LLMManager()

def generate_trade_recommendation_parallel(combined_data: Dict[str, Any], user_balance: Dict[str, Any]) -> str:
    """Generate trade recommendation using parallel processing for data analysis"""
    
    # Use threading to parallelize the processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        # First thread: Summarize data
        summarize_thread = executor.submit(summarize_pair_data, combined_data)
        
        # Second thread: Calculate technical indicators (most intensive)
        indicators_thread = executor.submit(calculate_technical_indicators, combined_data)
        
        # Wait for results
        summarized_data = summarize_thread.result()
        technical_indicators = indicators_thread.result()
    
    # Format pair information for the LLM
    pair = combined_data.get("pair", "Unknown")
    exchange = combined_data.get("exchange", "Unknown")
    timeframe = combined_data.get("timeframe", "Unknown")
    pair_info = f"{pair} on {exchange} ({timeframe})"
    
    # Format the technical indicators for LLM consumption and add smart money context
    formatted_indicators = format_indicators_for_llm(technical_indicators)
    enriched_indicators = add_smart_money_context(formatted_indicators)
    
    # Create the system prompt with enhanced context for smart money concepts
    system_prompt = f"""You are an expert trading advisor with deep knowledge of technical analysis, market dynamics, and institutional behavior patterns.
    
Analyze the provided market data and technical indicators for {pair_info} and generate a trading recommendation in JSON format.

Pay special attention to the Smart Money Concepts (SMC) indicators that reveal institutional activity:
1. Liquidity sweeps - when price briefly breaks a significant level then reverses (sign of manipulation)
2. Order blocks - zones where significant institutional orders led to strong directional moves
3. Fair value gaps - imbalances in price that tend to get filled as price returns to establish fair value
4. Delta divergences - contradictions between price movement and underlying buying/selling pressure
5. Market structure shifts - changes in the sequence of highs and lows that indicate trend changes

Your recommendation should follow this structure:
{{
    "pair": "{pair_info}",
    "signal": "buy/sell/hold",
    "trade_size": float,
    "take_profit": {{
        "level1": price,
        "level2": price,
        "level3": price
    }},
    "stop_loss": {{
        "level": price
    }},
    "confidence": "high/medium/low",
    "explanation": "detailed explanation of your recommendation",
    "timeframe": "short_term/medium_term/long_term",
    "key_levels": {{
        "support_levels": [level1, level2],
        "resistance_levels": [level1, level2]
    }},
    "analysis_details": {{
        "technical_analysis": "summary of traditional indicators",
        "market_structure": "summary of market structure analysis",
        "smart_money_concepts": "analysis of institutional activity and liquidity patterns",
        "risk_assessment": "analysis of risk/reward",
        "advanced_patterns": "summary of harmonic patterns, elliott waves, and other advanced patterns detected"
    }}
}}

Calculate appropriate trade size based on user balance information:
- For volatile or risky assets, recommend using 1-3% of available balance
- For less volatile or more established assets, recommend using 3-5% of available balance
- Never recommend using more than 5% of available balance for a single trade

For the take profit levels, calculate based on key technical levels or percentage targets (such as 1.5%, 3%, and 7% for short-term trades).
For the stop loss, recommend a level that respects recent support levels and volatility (typically ATR-based).

Be specific with price levels. Avoid vague recommendations. Use the actual pricing data provided.
    """
    
    user_balance_info = f"User balance information: {json.dumps(user_balance, default=str)}"
    
    # Create safe technical indicators string
    technical_indicators_str = "Technical Indicators: Not available or error in processing."
    try:
        # Use the new enriched indicators
        technical_indicators_str = f"Technical Indicators:\n{json.dumps(enriched_indicators, default=str)}"
    except Exception as e:
        logger.error(f"Error formatting technical indicators: {str(e)}")
    
    messages = [
        SystemMessage(content=system_prompt),
        UserMessage(content=f"Market Summary:\n{summarized_data}\n\n{technical_indicators_str}\n\n{user_balance_info}")
    ]
    
    try:
        # Get LLM response
        recommendation = llm_manager.get_response(
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        if not recommendation:
            logger.error("Empty response from LLM")
            return json.dumps({
                "error": "Empty response from recommendation engine",
                "pair": pair_info,
                "signal": "hold",
                "explanation": "Unable to generate recommendation due to an empty response from the AI system."
            })
        
        # Format the JSON result
        formatted_result = format_json_result(recommendation)
        
        # Return either the formatted result or raw response if formatting fails
        if formatted_result:
            result = formatted_result[0] if isinstance(formatted_result, list) else formatted_result
            return json.dumps(result)
        else:
            logger.warning(f"Failed to format recommendation as JSON: {recommendation[:100]}...")
            # Create a fallback JSON response
            fallback_response = {
                "pair": pair_info,
                "signal": "hold",
                "explanation": "Unable to generate proper recommendation format. Please try again.",
                "trade_size": 0.0,
                "take_profit": {"level1": 0.0},
                "stop_loss": {"level": 0.0},
                "confidence": "low",
                "timeframe": combined_data.get('timeframe', 'short_term'),
                "key_levels": {},
                "analysis_details": {"error": "Formatting failed"}
            }
            return json.dumps(fallback_response)
            
    except Exception as e:
        logger.error(f"Error generating trade recommendation: {str(e)}")
        return json.dumps({
            "error": f"Error generating recommendation: {str(e)}",
            "pair": pair_info
        }) 