"""
Trade Recommendation Engine
Contains functions for generating trading recommendations using LLM
"""
import logging
import json
from typing import Dict, Any, List, Union
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.ai.inference.models import SystemMessage, UserMessage

from app.services.llm_manager import LLMManager
from app.utils.json_utils import format_json_result
from app.core.analysis_engine import summarize_pair_data, format_indicators_for_llm, add_smart_money_context, optimize_indicators_for_llm

# Set up logger
logger = logging.getLogger(__name__)

# Initialize LLM manager
llm_manager = LLMManager()

def generate_trade_recommendation_parallel(combined_data: Dict[str, Any], user_balance: Dict[str, Any]) -> str:
    """Generate trade recommendation using parallel processing for data analysis"""
    
    # Prepare data for different analysis types
    market_data = combined_data.get('market_data', {})
    technical_indicators = combined_data.get('technical_indicators', {})
    pair_info = combined_data.get('pair', '')
    
    # Create a summarized view of the data for the LLM
    try:
        summarized_data = summarize_pair_data(market_data)
    except Exception as e:
        logger.error(f"Error summarizing market data: {str(e)}")
        summarized_data = f"Error summarizing market data for {pair_info}."
    
    # Sanitize and optimize indicators for LLM consumption
    try:
        # First optimize to remove null values and reduce token usage
        optimized_indicators = optimize_indicators_for_llm(technical_indicators)
        
        # Then format in a structured, LLM-friendly format
        formatted_indicators = format_indicators_for_llm(optimized_indicators)
        
        # Finally, add contextual information to help the LLM interpret the data
        enriched_indicators = add_smart_money_context(formatted_indicators)
        
        # Use the enriched indicators instead of raw technical indicators
        technical_indicators = enriched_indicators
    except Exception as e:
        logger.error(f"Error formatting indicators for LLM: {str(e)}")
    
    # Create the final message for comprehensive analysis
    system_prompt = f"""
You are an expert trading analyst specialized in cryptocurrency and forex markets. Analyze the provided market data and generate a detailed trading recommendation for {pair_info}.

Your analysis should consider technical indicators, market structure, and potential risks.

Pay special attention to the following advanced market analysis elements if available:

1. Volume Profile - Identify if price is near high-volume nodes (HVNs) which act as magnets/areas of support/resistance, or low-volume nodes (LVNs) which price tends to move through quickly. The Point of Control (POC) represents the highest volume price level and is significant.

2. Harmonic Patterns - If any harmonic patterns (Gartley, Butterfly, Bat) have been detected, evaluate their completion level and pattern quality. Higher confidence patterns (>75%) should be weighted more heavily in your analysis.

3. Divergences - Check for regular and hidden divergences in RSI and MACD. Regular divergences indicate potential reversals, while hidden divergences confirm the current trend. Consider the strength of divergences in your analysis.

4. Elliott Wave Patterns - If Elliott Wave patterns have been detected, consider the current position in the wave cycle and confidence level of the pattern. Incorporate this into your overall trend analysis.

5. Mean Reversion Index - Utilize this indicator to identify potential overbought or oversold conditions that may lead to mean reversion opportunities. Values above +50 suggest price is likely to revert downward, while values below -50 suggest price is likely to revert upward.

6. Heikin-Ashi Analysis - Consider the Heikin-Ashi trend information for a smoothed view of the market direction. This can help filter out market noise and provide clearer trend signals than traditional candlesticks.

7. Order Flow Analysis - If bid/ask volume data is available, examine buying/selling pressure, cumulative delta, and absorption zones to identify potential institutional activity and smart money movements.

8. Stochastic RSI - Use this indicator to identify overbought/oversold conditions with greater sensitivity than regular RSI. Look for signal line crossovers as potential entry/exit triggers.

Base your response on facts from the data provided, not general market knowledge or assumptions.

Format your response as a JSON object with the following structure:
```json
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
        "technical_analysis": "summary",
        "market_structure": "summary",
        "risk_assessment": "summary",
        "advanced_patterns": "summary of harmonic patterns, elliott waves, and other advanced patterns detected"
    }}
}}
```

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
        technical_indicators_str = f"Technical Indicators:\n{json.dumps(technical_indicators, default=str)}"
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
        error_response = {
            "error": f"Failed to generate recommendation: {str(e)}",
            "pair": pair_info,
            "signal": "hold",
            "trade_size": 0.0,
            "explanation": "An error occurred while generating the recommendation."
        }
        return json.dumps(error_response)

def get_user_balances(dust_threshold: Decimal = Decimal('1.00')) -> List[Dict[str, Union[str, Decimal]]]:
    """Fetch user balances using parallel processing"""
    from app.services.coinbase_manager import CoinbaseManager
    coinbase_manager = CoinbaseManager()
    
    try:
        accounts = coinbase_manager.get_accounts()
        balances = []
        
        def process_account(account):
            try:
                if account is None:
                    return None
                    
                # Handle dictionary format
                if isinstance(account, dict):
                    currency = account.get('currency', '')
                    if 'available_balance' in account and isinstance(account['available_balance'], dict):
                        balance = Decimal(account['available_balance'].get('value', '0'))
                    else:
                        return None
                # Handle object format
                elif hasattr(account, 'currency') and hasattr(account, 'available_balance'):
                    currency = account.currency
                    if hasattr(account.available_balance, 'value'):
                        balance = Decimal(account.available_balance.value)
                    else:
                        return None
                else:
                    return None
                
                if balance >= dust_threshold:
                    return {
                        'currency': currency,
                        'available': float(balance),
                        'hold': 0.0,  # Default value
                        'total': float(balance),
                        'usd_value': float(balance) if currency == 'USD' else 0.0
                    }
            except Exception as e:
                logger.error(f"Error processing account: {str(e)}")
            return None
        
        # Process accounts based on the format received
        if isinstance(accounts, dict) and 'accounts' in accounts:
            # Dictionary format
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_account, account) 
                          for account in accounts['accounts']]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            balances.append(result)
                    except Exception as e:
                        logger.error(f"Error processing balance: {str(e)}")
        else:
            # Possibly object format or list
            account_list = accounts
            if hasattr(accounts, 'accounts'):
                account_list = accounts.accounts
                
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_account, account) 
                          for account in account_list if account is not None]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            balances.append(result)
                    except Exception as e:
                        logger.error(f"Error processing balance: {str(e)}")
        
        return balances
    
    except Exception as e:
        logger.error(f"Error fetching user balances: {str(e)}")
        return []

def generate_trade_recommendation(combined_data: str, user_balance: Dict[str, Any]) -> str:
    """
    Generate trade recommendation using LLM
    
    Args:
        combined_data: JSON string containing market data and analysis
        user_balance: Dictionary with user balance information
    
    Returns:
        Raw string response from LLM containing trade recommendation
    """
    # Parse the combined_data JSON string
    data = json.loads(combined_data)
    # Get the first pair's data (assumes single pair analysis)
    pair_data = next(iter(data.values()))
    
    # Properly handle current_price extraction with a default value
    current_price = pair_data.get('current_price', 0)
    if current_price is None:
        logger.warning("Current price is None in generate_trade_recommendation. Setting to default.")
        current_price = 1.0  # Default value to prevent multiplication errors
    
    signals_query = f"""
    **SYSTEM**  
    Analyze market data and output JSON trade signals without explanations. Current Price: {current_price}

    **Price Rules**  
    1. SELL Signals:  
       - TP: [{current_price*0.98}, {current_price*0.96}, {current_price*0.94}]  
       - SL: {current_price*1.02}  
       - Breakeven: {current_price}  
       - Trailing: {current_price*0.01}  

    2. BUY Signals:  
       - TP: [{current_price*1.02}, {current_price*1.04}, {current_price*1.06}]  
       - SL: {current_price*0.98}  
       - Breakeven: {current_price}  
       - Trailing: {current_price*0.01}  

    **validation**  
    - Confirm:  
      1. TP direction vs entry  
      2. SL opposite TP  
      3. ATR validation  
      4. Signal-target alignment  
      5. SL < current_price  

    **analysis**  
    1. Trend: EMA(9,12,21,26,50), Market Structure, Ichimoku, Heikin-Ashi, Supertrend
    2. Momentum: RSI, MACD, Stochastic, Stochastic RSI, CCI, ADX(>25), Mean Reversion Index  
    3. Volume: OBV, ADL, MFI(14,21,50), CMF, Klinger, Volume Profile, Order Flow (if available)
    4. Patterns: Harmonic Patterns, Elliott Wave Patterns, Divergences
    5. Targets: Swing levels, S/R zones, Fib(1.618,2.618), VPOC, HVN/LVN  
    6. Risk:  
       - SL: ATR × [1.5|2|3] (ADX <20|20-40|>40)  
       - RR: 1:3(LOW), 1:2.5(MED), 1:2(HIGH)  
       - Size: (Account×1%)/(Entry-SL)  

    **Output Rules**  
    - Numerical values only (no text/symbols)
    - DO NOT include comments in your JSON (no // comments)
    - DO NOT include explanations inside the JSON values
    - Example valid format:  
      "take_profit": {{"tp1":45000.5,"tp2":46000.25,"tp3":47000.75}}  

     **JSON Structure**  
    {{
        "pair": "SYMBOL-USD",
        "signal": "BUY" or "SELL"  or "WAIT",
        "trade_size": [calculate: (suggested_position_size * 0.95) for safety],
        "take_profit": {{
            "tp1": [nearest major resistance],
            "tp2": [next significant resistance],
            "tp3": [fibonacci extension target]
        }},
        "stop_loss": {{
            "initial": [ATR-based stop],
            "breakeven": [entry + (ATR * 1)],
            "trailing": [ATR-based trailing parameters]
        }},
        "confidence": "LOW" or "MEDIUM" or "HIGH",
        "explanation": "Concise justification < 150 words)",
        "timeframe": [from data],
        "key_levels": {{
            "support_levels": [
                {{
                    "price": [level],
                    "strength": "weak/medium/strong",
                    "type": "swing low/demand zone/fibonacci"
                }}
            ],
            "resistance_levels": [
                {{
                    "price": [level],
                    "strength": "weak/medium/strong",
                    "type": "swing high/supply zone/fibonacci"
                }}
            ]
        }},
        "advanced_patterns": {{
            "harmonic_patterns": [pattern type and confidence],
            "elliott_wave": [current position in wave cycle],
            "divergences": [type and strength of divergences]
        }}
    }}

    market data:  
    {combined_data}
    """
    
    messages = [
        SystemMessage(content="You are a cryptocurrency trading analyst focused on risk-managed trading signals. Return valid JSON without comments or explanations inside the JSON structure."),
        UserMessage(content=signals_query)
    ]
    
    # Return the raw response from LLM without any processing
    return llm_manager.get_response(messages, temperature=0.7, max_tokens=4096) 