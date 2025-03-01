"""
API Routes
Contains all FastAPI route handlers
"""
import uuid
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
from decimal import Decimal

from app.models.schemas import AnalysisRequest, TradeRequest, TradeResponse
from app.services.coinbase_manager import CoinbaseManager
from app.core.data_fetcher import get_comprehensive_pair_data, get_timeframe_params
from app.core.analysis_engine import calculate_technical_indicators, summarize_pair_data
from app.core.trade_recommendation import generate_trade_recommendation, get_user_balances
from app.utils.json_utils import format_decimal, format_json_result, DecimalEncoder

from datetime import datetime, timedelta, timezone
import json
import numpy as np
import pandas as pd
import math

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize services
coinbase_manager = CoinbaseManager()

@router.post("/execute-trade")
async def execute_trade(trade: TradeRequest):
    """Execute a trade order with comprehensive error handling and validation"""
    try:
        # Generate a unique client order ID with timestamp
        client_order_id = f"trade_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        
        # Validate inputs
        if not trade.product_id or not trade.side or not trade.order_type:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: product_id, side, and order_type are required"
            )
            
        # Handle different order types
        result = None
        if trade.order_type.lower() == "market":
            if trade.side.lower() == "buy":
                if not trade.quote_size and not trade.base_size:
                    raise HTTPException(
                        status_code=400,
                        detail="Either quote_size or base_size must be provided for market buy orders"
                    )
                
                # Create kwargs based on provided parameters
                kwargs = {}
                if trade.quote_size:
                    kwargs["quote_size"] = trade.quote_size
                elif trade.base_size:
                    kwargs["base_size"] = trade.base_size
                
                result = coinbase_manager.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=trade.product_id,
                    **kwargs
                )
            elif trade.side.lower() == "sell":
                if not trade.base_size:
                    raise HTTPException(
                        status_code=400,
                        detail="base_size must be provided for market sell orders"
                    )
                
                result = coinbase_manager.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=trade.product_id,
                    base_size=trade.base_size
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid side parameter. Must be 'buy' or 'sell'."
                )
        elif trade.order_type.lower() == "limit":
            if not trade.base_size or not trade.limit_price:
                raise HTTPException(
                    status_code=400,
                    detail="base_size and limit_price must be provided for limit orders"
                )
                
            post_only = trade.self_trade_prevention_id == "POST_ONLY"
            
            if trade.side.lower() == "buy":
                result = coinbase_manager.limit_order_gtc_buy(
                    client_order_id=client_order_id,
                    product_id=trade.product_id,
                    base_size=trade.base_size,
                    limit_price=trade.limit_price,
                    post_only=post_only
                )
            elif trade.side.lower() == "sell":
                result = coinbase_manager.limit_order_gtc_sell(
                    client_order_id=client_order_id,
                    product_id=trade.product_id,
                    base_size=trade.base_size,
                    limit_price=trade.limit_price,
                    post_only=post_only
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid side parameter. Must be 'buy' or 'sell'."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid order_type parameter. Must be 'market' or 'limit'."
            )
        
        # Process and return the result
        order_details = {
            "success": True,
            "order_id": result.order_id if hasattr(result, 'order_id') else None,
            "client_order_id": client_order_id,
            "product_id": trade.product_id,
            "side": trade.side,
            "order_type": trade.order_type,
            "status": "pending" if not hasattr(result, 'status') else result.status
        }
        
        if trade.order_type.lower() == "limit" and trade.limit_price:
            order_details["limit_price"] = trade.limit_price
            
        if trade.base_size:
            order_details["base_size"] = trade.base_size
            
        if trade.quote_size:
            order_details["quote_size"] = trade.quote_size
            
        return order_details
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute trade: {str(e)}"
        )

@router.post("/analyze", response_model=TradeResponse)
async def analyze_trade(request: AnalysisRequest):
    """Generate trade analysis and recommendations"""
    try:
        logger.info(f"Analyzing trade for {request.pair} on {request.timeframe} timeframe with amount: {request.amount}")
        
        # Get market data
        pair_data = get_comprehensive_pair_data(request.pair)
            
        # Calculate technical indicators
        indicators = calculate_technical_indicators(pair_data)
        
        # Log the keys of the indicators dictionary for debugging
        logger.info(f"Available indicator keys: {list(indicators.keys())}")
        
        # Helper function to recursively sanitize and filter out None values
        def sanitize_value(value):
            # For None values, return None to be filtered out later
            if value is None:
                return None
            # Convert numpy scalar types to Python native types
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                # Check for NaN and return None instead
                if np.isnan(value):
                    return None
                return float(value)
            elif isinstance(value, np.ndarray):
                # Convert numpy array to list and filter out None/NaN values
                result = value.tolist()
                # Filter out None and NaN values
                filtered_result = []
                for item in result:
                    if item is None:
                        continue
                    if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
                        continue
                    filtered_result.append(item)
                return filtered_result if filtered_result else None
            # Handle dictionary values recursively
            elif isinstance(value, dict):
                result = {}
                for k, v in value.items():
                    sanitized_v = sanitize_value(v)
                    # Only include non-None values in the result
                    if sanitized_v is not None:
                        result[k] = sanitized_v
                return result if result else None  # Return None for empty dictionaries
            # Handle list values recursively
            elif isinstance(value, (list, tuple)):
                result = [sanitize_value(item) for item in value]
                # Filter out None values from the list
                result = [item for item in result if item is not None]
                return result if result else None  # Return None for empty lists
            # For complex objects, convert to string
            elif not isinstance(value, (int, float, bool, str, list, dict, type(None))):
                return str(value)
            # Handle float NaN/Inf in native Python
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            # Return other primitive values as they are
            else:
                return value
        
        # Apply sanitization to all indicators
        sanitized_indicators = {}
        for key, value in indicators.items():
            sanitized_value = sanitize_value(value)
            if sanitized_value is not None:  # Only include non-None values
                sanitized_indicators[key] = sanitized_value
        
        # Get market summary
        market_summary = summarize_pair_data(pair_data)
        
        # Get current price for position sizing
        current_price = None
        product_details = pair_data.get('product_details', {})
        if isinstance(product_details, dict) and 'price' in product_details:
            current_price = float(product_details['price'])
        elif hasattr(product_details, 'price'):
            current_price = float(product_details.price)
        
        # Fallback: Use midpoint of best bid/ask if price is not available
        if current_price is None:
            best_bid_ask = pair_data.get('best_bid_ask', {})
            bid = ask = None
            
            # Handle both dictionary and object formats
            if isinstance(best_bid_ask, dict):
                bid = float(best_bid_ask.get('bid', 0)) if 'bid' in best_bid_ask else None
                ask = float(best_bid_ask.get('ask', 0)) if 'ask' in best_bid_ask else None
            elif hasattr(best_bid_ask, 'bid') and hasattr(best_bid_ask, 'ask'):
                bid = float(best_bid_ask.bid) if best_bid_ask.bid else None
                ask = float(best_bid_ask.ask) if best_bid_ask.ask else None
            
            # Calculate midpoint if both bid and ask are available
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                current_price = (bid + ask) / 2
                logger.info(f"Using bid-ask midpoint for price: {current_price}")
        
        # Second fallback: Use the latest candle close price if available
        if current_price is None and pair_data.get('candles') and len(pair_data['candles']) > 0:
            try:
                # Get the most recent candle (last in the list)
                latest_candle = pair_data['candles'][-1]
                if isinstance(latest_candle, dict) and 'close' in latest_candle:
                    current_price = float(latest_candle['close'])
                    logger.info(f"Using latest candle close price for {request.pair}: {current_price}")
            except Exception as e:
                logger.warning(f"Error getting price from latest candle: {str(e)}")
        
        # Log the current price value
        logger.info(f"Current price for {request.pair}: {current_price}")
        
        # Final fallback: Set a default value if all methods fail
        if current_price is None:
            logger.warning(f"Unable to determine current price for {request.pair} using any method, using default value 1.0")
            current_price = 1.0
        
        # Calculate position size if amount is provided
        position_size = None
        if request.amount and current_price:
            try:
                position_size = float(request.amount) / current_price
                # Format to appropriate decimal places
                position_size = format_decimal(position_size)
                logger.info(f"Calculated position size: {position_size} from amount: {request.amount}")
            except Exception as e:
                logger.warning(f"Error calculating position size: {str(e)}")
        
        # Combine data
        combined_data = {
            request.pair: {
                'indicators': sanitized_indicators,
                'summary': market_summary,
                'timeframe': request.timeframe,
                'amount': request.amount,
                'current_price': current_price,
                'suggested_position_size': position_size
            }
        }
        
        # Get user balance for position sizing
        user_balance = get_user_balances()
        
        # Generate trade recommendation
        recommendation = generate_trade_recommendation(json.dumps(combined_data, cls=DecimalEncoder), user_balance)
        
        # Parse and format recommendation
        formatted_recommendation = format_json_result(recommendation)
        if not formatted_recommendation:
            raise HTTPException(status_code=500, detail="Failed to generate trade recommendation")
        
        # Get the first recommendation if it's a list
        result = formatted_recommendation[0] if isinstance(formatted_recommendation, list) else formatted_recommendation
        
        # Ensure required fields exist in the result
        if 'pair' not in result:
            result['pair'] = request.pair
            
        if 'signal' not in result:
            result['signal'] = 'WAIT'
            
        if 'confidence' not in result:
            result['confidence'] = 'LOW'
            
        if 'explanation' not in result:
            result['explanation'] = 'Analysis based on comprehensive technical indicators and market conditions.'
            
        if 'timeframe' not in result:
            result['timeframe'] = request.timeframe
            
        # Handle missing take_profit
        if 'take_profit' not in result:
            # Create default take_profit based on current price
            if current_price:
                result['take_profit'] = {
                    'tp1': str(current_price * 1.01),
                    'tp2': str(current_price * 1.02),
                    'tp3': str(current_price * 1.03)
                }
            else:
                result['take_profit'] = {'tp1': '0', 'tp2': '0', 'tp3': '0'}
                
        # Handle missing stop_loss
        if 'stop_loss' not in result:
            # Create default stop_loss based on current price
            if current_price:
                result['stop_loss'] = {
                    'initial': str(current_price * 0.99),
                    'breakeven': str(current_price),
                    'trailing': str(current_price * 0.005)
                }
            else:
                result['stop_loss'] = {'initial': '0', 'breakeven': '0', 'trailing': '0'}
                
        # Handle missing key_levels
        if 'key_levels' not in result:
            result['key_levels'] = {
                'support_levels': [],
                'resistance_levels': []
            }
        
        # Use calculated position size if available, otherwise use recommended size or default
        if position_size and request.amount:
            result['trade_size'] = position_size
        elif 'trade_size' in result:
            # Ensure trade_size is properly formatted
            if isinstance(result.get('trade_size'), (list, tuple)):
                result['trade_size'] = str(result['trade_size'][0])
            else:
                result['trade_size'] = str(result['trade_size'])
        else:
            # Set a default trade_size if missing
            logger.warning("trade_size missing from LLM response, using default")
            if position_size:
                result['trade_size'] = position_size
            elif current_price and request.amount:
                # Calculate a default position size
                default_size = float(request.amount) / current_price if current_price > 0 else 0
                result['trade_size'] = format_decimal(default_size)
            else:
                result['trade_size'] = "0"
        
        # Extract data for market summary
        candles = pd.DataFrame(pair_data['candles']) if 'candles' in pair_data and pair_data['candles'] else pd.DataFrame()
        
        # Prepare structured market summary for frontend
        structured_market_summary = {
            'price_summary': {
                'last_price': float(candles['close'].iloc[-1]) if not candles.empty else 0,
                'price_high_24h': float(candles['high'].max()) if not candles.empty else 0,
                'price_low_24h': float(candles['low'].min()) if not candles.empty else 0,
                'price_average_24h': float(candles['close'].mean()) if not candles.empty else 0,
                'price_volatility': float(candles['close'].std()) if not candles.empty else 0
            },
            'volume_summary': {
                'volume_average_24h': float(candles['volume'].mean()) if not candles.empty else 0,
                'volume_highest_24h': float(candles['volume'].max()) if not candles.empty else 0,
                'volume_lowest_24h': float(candles['volume'].min()) if not candles.empty else 0
            },
            'market_sentiment': {
                'buy_sell_ratio': 1.0,  # Default value
                'dominant_side': 'BUY'  # Default value
            }
        }
        
        # Add analysis details
        result['analysis_details'] = {
            'technical_indicators': sanitized_indicators,
            'market_summary': structured_market_summary,
            'amount_usd': request.amount if request.amount is not None else 0.0,
            'current_price': current_price if current_price is not None else 0.0
        }
        
        # Helper function to safely convert to float
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Format the result to match the original code
        string_formatted_result = {
            key: (str(value) if isinstance(value, (Decimal, float)) and key == 'trade_size'
                  else {
                      'tp1': str(value.get('tp1', '0')),
                      'tp2': str(value.get('tp2', '0')),
                      'tp3': str(value.get('tp3', '0'))
                  } if isinstance(value, dict) and key == 'take_profit'
                  else {
                      'initial': str(value.get('initial', '0')),
                      'breakeven': str(value.get('breakeven', '0')),
                      'trailing': str(value.get('trailing', {}).get('distance', value.get('trailing', {}).get('trail', '0'))) 
                                    if isinstance(value.get('trailing'), dict) 
                                    else str(value.get('trailing', '0'))
                  } if isinstance(value, dict) and key == 'stop_loss'
                  else float(value) if isinstance(value, (np.integer, np.floating)) 
                  else value)
            for key, value in result.items()
        }
        
        # IMPORTANT - Make absolutely sure trade_size is in the result and is a string before conversion
        if 'trade_size' not in string_formatted_result:
            string_formatted_result['trade_size'] = "0"
        
        # Now convert the string values to float for the TradeResponse model
        final_result = {}
        for key, value in string_formatted_result.items():
            if key == 'trade_size':
                final_result[key] = safe_float(value)
            elif key == 'take_profit' and isinstance(value, dict):
                final_result[key] = {
                    'tp1': safe_float(value['tp1']),
                    'tp2': safe_float(value['tp2']),
                    'tp3': safe_float(value['tp3'])
                }
            elif key == 'stop_loss' and isinstance(value, dict):
                final_result[key] = {
                    'initial': safe_float(value['initial']),
                    'breakeven': safe_float(value['breakeven']),
                    'trailing': safe_float(value['trailing'])
                }
            else:
                final_result[key] = value
                
        # CRITICAL - Force ensure trade_size exists as a float in the final result
        if 'trade_size' not in final_result or not isinstance(final_result['trade_size'], float):
            if 'trade_size' in string_formatted_result:
                try:
                    final_result['trade_size'] = float(string_formatted_result['trade_size'])
                except (ValueError, TypeError):
                    final_result['trade_size'] = 0.0
            else:
                final_result['trade_size'] = 0.0
        
        # Final validation to ensure stop_loss.trailing is never invalid
        if isinstance(final_result.get('stop_loss'), dict) and (final_result['stop_loss'].get('trailing') is None or final_result['stop_loss'].get('trailing') == "None"):
            final_result['stop_loss']['trailing'] = 0.0
            
        # Final check to ensure all technical indicators are primitive types and null values are filtered
        if 'analysis_details' in final_result and 'technical_indicators' in final_result['analysis_details']:
            # Get the technical indicators from the analysis details
            raw_indicators = final_result['analysis_details']['technical_indicators']
            
            # Ensure we have all the key indicators explicitly present in the output
            if isinstance(raw_indicators, dict):
                # Check for ADX specifically since it's missing
                if 'ADX' not in raw_indicators and 'Market_Structure' in raw_indicators and 'adx' in raw_indicators['Market_Structure']:
                    # If ADX is in Market_Structure but not in the main dict, add it
                    raw_indicators['ADX'] = raw_indicators['Market_Structure']['adx']
                
                # Make sure other important indicators are exposed at the top level
                important_indicators = ['Plus_DI', 'Minus_DI', 'RSI', 'Stoch_K', 'Stoch_D', 'Williams_R']
                for indicator in important_indicators:
                    if indicator not in raw_indicators and 'Market_Structure' in raw_indicators:
                        if indicator.lower() in raw_indicators['Market_Structure']:
                            raw_indicators[indicator] = raw_indicators['Market_Structure'][indicator.lower()]
                
                # Apply our sanitization function again to ensure we filter out any null values
                # that might have been added after initial sanitization
                sanitized_indicators = {}
                for key, value in raw_indicators.items():
                    sanitized_value = sanitize_value(value)
                    if sanitized_value is not None:  # Only include non-None values
                        sanitized_indicators[key] = sanitized_value
                
                # Replace with the re-sanitized version
                final_result['analysis_details']['technical_indicators'] = sanitized_indicators
            
            # Log the keys for debugging
            logger.info(f"Technical indicators included in response: {list(final_result['analysis_details']['technical_indicators'].keys())}")
        
        logger.info(f"Generated analysis with trade size: {final_result['trade_size']} for amount: {request.amount}")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in analyze_trade: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/balance")
async def get_balance():
    """Get user's balance information"""
    try:
        balances = get_user_balances()
        return {"balances": balances}
    except Exception as e:
        logger.error(f"Error fetching balance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch balance: {str(e)}"
        )

@router.get("/market_data/{pair}")
async def get_market_data(pair: str):
    """Get comprehensive market data for a trading pair"""
    try:
        market_data = get_comprehensive_pair_data(pair)
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch market data: {str(e)}"
        )

@router.get("/orders")
async def get_orders():
    """Get user's orders"""
    try:
        orders = coinbase_manager.list_orders()
        
        formatted_orders = []
        if hasattr(orders, 'orders'):
            for order in orders.orders:
                formatted_order = {
                    "order_id": order.order_id,
                    "product_id": order.product_id,
                    "side": order.side,
                    "status": order.status,
                    "created_time": order.created_time,
                    "client_order_id": order.client_order_id
                }
                
                # Determine order type from order configuration
                if hasattr(order, 'order_configuration'):
                    config = order.order_configuration
                    
                    if hasattr(config, 'market_market_ioc'):
                        formatted_order["order_type"] = "market"
                        if hasattr(config.market_market_ioc, 'quote_size'):
                            formatted_order["quote_size"] = config.market_market_ioc.quote_size
                        elif hasattr(config.market_market_ioc, 'base_size'):
                            formatted_order["base_size"] = config.market_market_ioc.base_size
                    
                    elif hasattr(config, 'limit_limit_gtc'):
                        formatted_order["order_type"] = "limit"
                        formatted_order["base_size"] = config.limit_limit_gtc.base_size
                        formatted_order["limit_price"] = config.limit_limit_gtc.limit_price
                        formatted_order["post_only"] = config.limit_limit_gtc.post_only
                    
                    # Add more order types as needed (stop, etc.)
                    else:
                        # Default if configuration type cannot be determined
                        formatted_order["order_type"] = "unknown"
                else:
                    # Default if no configuration available
                    formatted_order["order_type"] = "unknown"
                
                # Add filled details if available
                if hasattr(order, 'filled_size'):
                    formatted_order["filled_size"] = format_decimal(order.filled_size)
                
                if hasattr(order, 'average_filled_price'):
                    formatted_order["average_filled_price"] = format_decimal(order.average_filled_price)
                
                formatted_orders.append(formatted_order)
        
        return {"orders": formatted_orders}
    except Exception as e:
        logger.error(f"Error fetching orders: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch orders: {str(e)}"
        )

@router.get("/candles/{product_id}")
async def get_candles(product_id: str, timeframe: str = "1d"):
    """Get candlestick data for a specific product with specified timeframe"""
    try:
        # Get candles with specified timeframe
        granularity, days_back = get_timeframe_params(timeframe)
        end_time = int(datetime.now(timezone.utc).timestamp())
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
        
        # Use get_public_candles instead of get_product_candles
        response = coinbase_manager.get_public_candles(
            product_id=product_id,
            start=start_time,
            end=end_time,
            granularity=granularity
        )
        
        if not response or not hasattr(response, 'candles'):
            raise HTTPException(
                status_code=404,
                detail=f"No candle data found for {product_id}"
            )

        # Format candles data
        formatted_candles = []
        for candle in response.candles:
            formatted_candles.append({
                'start': candle['start'],
                'low': format_decimal(candle['low']),
                'high': format_decimal(candle['high']),
                'open': format_decimal(candle['open']),
                'close': format_decimal(candle['close']),
                'volume': format_decimal(candle['volume'])
            })

        # Sort candles by time (oldest first)
        formatted_candles.sort(key=lambda x: x['start'])
        
        return {
            "candles": formatted_candles,
            "product_id": product_id,
            "timeframe": timeframe,
            "granularity": granularity
        }

    except Exception as e:
        logger.error(f"Error fetching candles for {product_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch candle data: {str(e)}"
        )

@router.get("/pairs")
async def get_available_pairs():
    """Get available trading pairs"""
    try:
        logger.info("Fetching available trading pairs...")
        response = coinbase_manager.get_products()
        logger.debug(f"Raw response from Coinbase: {response}")
        
        pairs = []
        if hasattr(response, 'products'):
            pairs = [
                product.product_id 
                for product in response.products
                if (product.quote_currency_id == "USD" and 
                    product.status == "online")
            ]
            logger.info(f"Found {len(pairs)} valid trading pairs")
            
        return {
            "pairs": sorted(pairs),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error fetching pairs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch trading pairs: {str(e)}"
        )

@router.get("/timeframes")
async def get_timeframes():
    """Get available timeframes"""
    return {
        "timeframes": [
            "1m", "5m", "15m", "30m", "1h", "4h", "1d"
        ]
    }

@router.get("/portfolio")
async def get_portfolio():
    """Get user's portfolio summary"""
    try:
        # Get accounts directly from coinbase manager
        accounts = coinbase_manager.get_accounts()
        portfolio = []
        
        for account in accounts['accounts']:
            balance = Decimal(account['available_balance']['value'])
            if balance > 0:
                portfolio.append({
                    'currency': account['currency'],
                    'balance': float(balance),
                    'value_usd': float(account['available_balance']['value']) 
                })
        
        return {
            "portfolio": portfolio,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch portfolio: {str(e)}"
        )

@router.get("/")
async def root():
    """Root endpoint returning API status"""
    return {
        "message": "Welcome to TradeGPT API",
        "version": "2.0.0",
        "status": "operational"
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Coinbase connection
        coinbase_manager.get_accounts()
        return {"status": "healthy", "services": {"coinbase": "connected"}}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        ) 