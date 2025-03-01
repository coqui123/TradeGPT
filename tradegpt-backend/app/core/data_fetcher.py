"""
Data Fetching Module
Contains functions for fetching market data
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from app.services.coinbase_manager import CoinbaseManager

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Coinbase manager
coinbase_manager = CoinbaseManager()

def get_timeframe_params(timeframe: str) -> tuple[str, int]:
    """Convert timeframe string to Coinbase granularity and calculate start time offset"""
    # Coinbase Advanced API granularity values
    granularity_map = {
        "1m": "ONE_MINUTE",
        "5m": "FIVE_MINUTE",
        "15m": "FIFTEEN_MINUTE",
        "30m": "THIRTY_MINUTE",
        "1h": "ONE_HOUR",
        "4h": "FOUR_HOUR",
        "1d": "ONE_DAY",
    }
    
    # Calculate how far back to look based on timeframe
    days_map = {
        "1m": 1,      # 1 day of 1-minute candles
        "5m": 3,      # 3 days of 5-minute candles
        "15m": 7,     # 7 days of 15-minute candles
        "30m": 10,    # 10 days of 30-minute candles
        "1h": 14,     # 14 days of hourly candles
        "4h": 30,     # 30 days of 4-hour candles
        "1d": 100,    # 100 days of daily candles (increased from 55)
    }
    
    if timeframe not in granularity_map:
        raise ValueError(f"Invalid timeframe: {timeframe}")
        
    return granularity_map[timeframe], days_map[timeframe]

def fetch_market_data(product_id: str, timeframe: str = "1d") -> Dict[str, Any]:
    """Fetch all market data for a product in parallel"""
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'product': executor.submit(coinbase_manager.get_product, product_id),
                'book': executor.submit(coinbase_manager.get_product_book, product_id, limit=100),
                'trades': executor.submit(coinbase_manager.get_market_trades, product_id, limit=100),
                'best_bid_ask': executor.submit(coinbase_manager.get_best_bid_ask, product_id)
            }
            
            # Get candles with specified timeframe
            granularity, days_back = get_timeframe_params(timeframe)
            end_time = int(datetime.now(timezone.utc).timestamp())
            start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
            
            futures['candles'] = executor.submit(
                coinbase_manager.get_public_candles,
                product_id,
                start=start_time,
                end=end_time,
                granularity=granularity
            )
            
            results = {}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching {name} for {product_id}: {str(e)}")
                    results[name] = None
            
            return results
    except Exception as e:
        logger.error(f"Error in fetch_market_data for {product_id}: {str(e)}")
        return {}

def get_comprehensive_pair_data(product_id: str):
    """
    Fetch comprehensive data for a specific trading pair using parallel requests
    """
    market_data = fetch_market_data(product_id)
    
    if not market_data:
        return {}
        
    pair_data = {}
    
    # Handle product details - could be object or dictionary
    product_details = market_data.get('product')
    if product_details:
        if isinstance(product_details, dict):
            pair_data['product_details'] = product_details
        else:
            # Convert object to dictionary if needed
            try:
                product_dict = {}
                if hasattr(product_details, '__dict__'):
                    # Some objects can be converted via __dict__
                    product_dict = {**product_details.__dict__}
                else:
                    # Otherwise try to get common attributes
                    if hasattr(product_details, 'product_id'):
                        product_dict['product_id'] = product_details.product_id
                    if hasattr(product_details, 'price'):
                        product_dict['price'] = str(product_details.price)  # Ensure price is a string
                    if hasattr(product_details, 'status'):
                        product_dict['status'] = product_details.status
                    if hasattr(product_details, 'base_increment'):
                        product_dict['base_increment'] = str(product_details.base_increment)
                    if hasattr(product_details, 'quote_increment'):
                        product_dict['quote_increment'] = str(product_details.quote_increment)
                    if hasattr(product_details, 'base_min_size'):
                        product_dict['base_min_size'] = str(product_details.base_min_size)
                    if hasattr(product_details, 'base_max_size'):
                        product_dict['base_max_size'] = str(product_details.base_max_size)
                    # Add other attributes as needed
                
                pair_data['product_details'] = product_dict
                logger.info(f"Extracted product details for {product_id}: {product_dict}")
            except Exception as e:
                logger.error(f"Error converting product details: {str(e)}")
                pair_data['product_details'] = {'error': 'Failed to process product details'}
    else:
        pair_data['product_details'] = {}
        
    # Handle order book - could be object or dictionary
    order_book = market_data.get('book')
    if order_book:
        if isinstance(order_book, dict):
            pair_data['order_book'] = order_book
        else:
            # Try to extract data from object
            book_dict = {}
            try:
                if hasattr(order_book, 'bids') and hasattr(order_book, 'asks'):
                    book_dict['bids'] = order_book.bids
                    book_dict['asks'] = order_book.asks
                # Add other attributes as needed
                
                pair_data['order_book'] = book_dict
            except Exception as e:
                logger.error(f"Error converting order book: {str(e)}")
                pair_data['order_book'] = {'error': 'Failed to process order book'}
    else:
        pair_data['order_book'] = {}
        
    # Handle recent trades
    trades = market_data.get('trades')
    if trades:
        if isinstance(trades, dict) or isinstance(trades, list):
            pair_data['recent_trades'] = trades
        else:
            # Try to extract data from object
            try:
                if hasattr(trades, 'trades'):
                    pair_data['recent_trades'] = trades.trades
                elif hasattr(trades, '__iter__'):
                    pair_data['recent_trades'] = list(trades)
                else:
                    pair_data['recent_trades'] = []
            except Exception as e:
                logger.error(f"Error converting trades: {str(e)}")
                pair_data['recent_trades'] = []
    else:
        pair_data['recent_trades'] = []
        
    # Handle best bid/ask
    best_bid_ask = market_data.get('best_bid_ask')
    if best_bid_ask:
        if isinstance(best_bid_ask, dict):
            pair_data['best_bid_ask'] = best_bid_ask
        else:
            bid_ask_dict = {}
            try:
                if hasattr(best_bid_ask, 'bid'):
                    bid_ask_dict['bid'] = best_bid_ask.bid
                if hasattr(best_bid_ask, 'ask'):
                    bid_ask_dict['ask'] = best_bid_ask.ask
                # Add other attributes as needed
                
                pair_data['best_bid_ask'] = bid_ask_dict
            except Exception as e:
                logger.error(f"Error converting best bid/ask: {str(e)}")
                pair_data['best_bid_ask'] = {'error': 'Failed to process best bid/ask'}
    else:
        pair_data['best_bid_ask'] = {}
    
    # Format candles data - handle both object and dictionary response formats
    candles_response = market_data.get('candles')
    formatted_candles = []
    
    if candles_response:
        try:
            # Handle object with candles attribute
            if hasattr(candles_response, 'candles'):
                candles_list = candles_response.candles
                for candle in candles_list:
                    if isinstance(candle, dict):
                        formatted_candles.append({
                            'start': candle.get('start', 0),
                            'low': float(candle.get('low', 0)),
                            'high': float(candle.get('high', 0)),
                            'open': float(candle.get('open', 0)),
                            'close': float(candle.get('close', 0)),
                            'volume': float(candle.get('volume', 0))
                        })
                    else:
                        # Handle object-style candle
                        if hasattr(candle, 'start') and hasattr(candle, 'low') and hasattr(candle, 'high') and \
                           hasattr(candle, 'open') and hasattr(candle, 'close') and hasattr(candle, 'volume'):
                            formatted_candles.append({
                                'start': candle.start,
                                'low': float(candle.low),
                                'high': float(candle.high),
                                'open': float(candle.open),
                                'close': float(candle.close),
                                'volume': float(candle.volume)
                            })
            # Handle dict with candles key
            elif isinstance(candles_response, dict) and 'candles' in candles_response:
                for candle in candles_response['candles']:
                    formatted_candles.append({
                        'start': candle.get('start', 0),
                        'low': float(candle.get('low', 0)),
                        'high': float(candle.get('high', 0)),
                        'open': float(candle.get('open', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('volume', 0))
                    })
            # Handle direct list of candles
            elif isinstance(candles_response, list):
                for candle in candles_response:
                    if isinstance(candle, dict):
                        formatted_candles.append({
                            'start': candle.get('start', 0),
                            'low': float(candle.get('low', 0)),
                            'high': float(candle.get('high', 0)),
                            'open': float(candle.get('open', 0)),
                            'close': float(candle.get('close', 0)),
                            'volume': float(candle.get('volume', 0))
                        })
        except Exception as e:
            logger.error(f"Error processing candles data: {str(e)}")
    
    pair_data['candles'] = formatted_candles
    
    return pair_data 