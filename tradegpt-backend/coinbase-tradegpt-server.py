import os
import uuid
from dotenv import load_dotenv
from coinbase.rest import RESTClient
from decimal import Decimal, getcontext
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Union, Literal, Optional
from datetime import datetime, timedelta, timezone
import statistics
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Json
import time

# Set decimal precision
getcontext().prec = 8

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TradeGPT API",
    description="AI-powered crypto trading analysis API using Azure and Coinbase",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder for Decimal objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, bool):  # Add handling for boolean values
            return bool(obj)
        return super(DecimalEncoder, self).default(obj)

class CoinbaseManager:
    """Coinbase Manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CoinbaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Load API credentials
        self.api_key = os.getenv('COINBASE_API_KEY')
        self.api_secret = os.getenv('COINBASE_API_SECRET')
        
        # Initialize client
        self.client = self._create_client()
        
    def _create_client(self) -> RESTClient:
        """Create a Coinbase REST client instance"""
        return RESTClient(
            api_key=self.api_key,
            api_secret=self.api_secret
        )
    
    def get_product(self, product_id: str):
        """Get product details"""
        return self.client.get_product(product_id)
    
    def get_product_book(self, product_id: str, limit: int = 100):
        """Get product order book"""
        return self.client.get_product_book(product_id, limit=limit)
    
    def get_market_trades(self, product_id: str, limit: int = 100):
        """Get market trades"""
        return self.client.get_market_trades(product_id, limit=limit)
    
    def get_best_bid_ask(self, product_id: str):
        """Get best bid/ask"""
        return self.client.get_best_bid_ask(product_id)
    
    def get_public_candles(self, product_id: str, start: int, end: int, granularity: str):
        """Get public candles data"""
        return self.client.get_public_candles(
            product_id=product_id,
            start=start,
            end=end,
            granularity=granularity
        )
    
    def get_accounts(self):
        """Get user accounts"""
        return self.client.get_accounts()
    
    def get_products(self):
        """Get available products"""
        return self.client.get_products()
    
    def market_order_buy(self, client_order_id: str, product_id: str, **kwargs):
        """Place market buy order"""
        return self.client.market_order_buy(
            client_order_id=client_order_id,
            product_id=product_id,
            **kwargs
        )
    
    def market_order_sell(self, client_order_id: str, product_id: str, base_size: str):
        """Place market sell order"""
        return self.client.market_order_sell(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=base_size
        )
    
    def limit_order_gtc_buy(self, client_order_id: str, product_id: str, base_size: str, limit_price: str, post_only: bool = False):
        """Place limit buy order"""
        return self.client.limit_order_gtc_buy(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=base_size,
            limit_price=limit_price,
            post_only=post_only
        )
    
    def limit_order_gtc_sell(self, client_order_id: str, product_id: str, base_size: str, limit_price: str, post_only: bool = False):
        """Place limit sell order"""
        return self.client.limit_order_gtc_sell(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=base_size,
            limit_price=limit_price,
            post_only=post_only
        )
    
    def list_orders(self, limit: int = 250, sort: str = "DESC", user_native_currency: str = "USD"):
        """List orders"""
        return self.client.list_orders(
            limit=limit,
            sort=sort,
            user_native_currency=user_native_currency
        )

class LLMManager:
    """LLM Manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.endpoint = os.getenv('LLM_API_ENDPOINT')
        self.model_name = os.getenv('LLM_API_MODEL_NAME')
        self.api_token = os.getenv('LLM_API_TOKEN')
        self.client = self.create_llm_client()
        
    def create_llm_client(self) -> ChatCompletionsClient:
        """Create a ChatCompletionsClient instance."""
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_token}"
        }
        
        return ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_token),
            headers=headers,
            api_version="2024-02-15-preview"
        )
    
    def get_response(self, messages: List, temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 1000) -> str:
        """Get the response from the ChatCompletionsClient."""
        try:
            response = self.client.complete(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                model=self.model_name
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise

llm_manager = LLMManager()
coinbase_manager = CoinbaseManager()

class AnalysisRequest(BaseModel):
    """Analysis Request"""
    pair: str
    timeframe: str
    amount: Optional[float] = None
    strategy: Optional[str] = "default"

class TradeRequest(BaseModel):
    """Trade Request"""
    product_id: str
    side: str
    order_type: str
    base_size: str
    limit_price: Optional[str] = None
    quote_size: Optional[str] = None
    self_trade_prevention_id: Optional[str] = None
    retail_portfolio_id: Optional[str] = None

class TradeResponse(BaseModel):
    """Trade Response"""
    pair: str
    signal: str
    trade_size: float
    take_profit: Dict[str, float] 
    stop_loss: Dict[str, float]    
    confidence: str
    explanation: str
    timeframe: str
    key_levels: dict
    analysis_details: dict

    class Config:
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            pd.Series: lambda x: x.tolist(),
            Decimal: str
        }

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
        "1d": 55,     # 55 days of daily candles
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
                    print(f"Error fetching {name} for {product_id}: {str(e)}")
                    results[name] = None
            
            return results
    except Exception as e:
        print(f"Error in fetch_market_data for {product_id}: {str(e)}")
        return {}

def get_comprehensive_pair_data(product_id: str):
    """
    Fetch comprehensive data for a specific trading pair using parallel requests
    """
    market_data = fetch_market_data(product_id)
    
    if not market_data:
        return {}
        
    pair_data = {
        'product_details': market_data.get('product', {}),
        'order_book': market_data.get('book', {}),
        'recent_trades': market_data.get('trades', {}),
        'best_bid_ask': market_data.get('best_bid_ask', {}),
    }
    
    # Format candles data - handle the response object directly
    candles_response = market_data.get('candles')
    formatted_candles = []
    
    if candles_response and hasattr(candles_response, 'candles'):
        for candle in candles_response.candles:
            formatted_candles.append({
                'start': candle['start'],
                'low': float(candle['low']),
                'high': float(candle['high']),
                'open': float(candle['open']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            })
    
    pair_data['candles'] = formatted_candles
    
    return pair_data

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

def klinger_oscillator(high, low, close, volume, short_period=34, long_period=55, signal_period=13):
    """Calculate Klinger Oscillator"""
    sv = volume * (2 * ((close - low) - (high - close)) / (high - low))
    sv = sv.fillna(0)
    ema_short = sv.ewm(span=short_period, adjust=False).mean()
    ema_long = sv.ewm(span=long_period, adjust=False).mean()
    kvo = ema_short - ema_long
    kvo_signal = kvo.ewm(span=signal_period, adjust=False).mean()
    return kvo, kvo_signal

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
    """Calculate Chaikin Money Flow (CMF)"""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0.0)  # Handle division by zero
    mfv = mfm * volume
    cmf = mfv.rolling(window).sum() / volume.rolling(window).sum()
    return cmf

def parabolic_sar(high, low, step=0.02, max_step=0.2):
    """Calculate Parabolic SAR"""
    sar = low[0]
    ep = high[0]
    af = step
    bull = True
    sar_list = [sar]

    for i in range(1, len(high)):
        if bull:
            sar = sar + af * (ep - sar)
        else:
            sar = sar - af * (sar - ep)
        
        bull_now = sar < low[i]
        
        if bull_now and not bull:
            bull = True
            sar = min(low[i-1], low[i])
            ep = high[i]
            af = step
        elif not bull_now and bull:
            bull = False
            sar = max(high[i-1], high[i])
            ep = low[i]
            af = step

        if bull:
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        else:
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)

        sar_list.append(sar)

    return pd.Series(sar_list, index=high.index)

def money_flow_index(high, low, close, volume, window=14):
    """Calculate Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = []
    negative_flow = []

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            negative_flow.append(money_flow[i])
            positive_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=window).sum()

    mfi = 100 * positive_mf / (positive_mf + negative_mf)
    return mfi

def percentage_price_oscillator(close, fast_period=12, slow_period=26, signal_period=9):
    """Calculate Percentage Price Oscillator"""
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    ppo = ((fast_ema - slow_ema) / slow_ema) * 100
    signal = ppo.ewm(span=signal_period, adjust=False).mean()
    histogram = ppo - signal
    return ppo, signal, histogram

def donchian_channels(high, low, window=20):
    """Calculate Donchian Channels"""
    upper = high.rolling(window=window).max()
    lower = low.rolling(window=window).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def rate_of_change(close, window=14):
    """Calculate Rate of Change"""
    return ((close / close.shift(window)) - 1) * 100

def commodity_channel_index(high, low, close, window=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def awesome_oscillator(high, low, short_period=5, long_period=34):
    """Calculate Awesome Oscillator"""
    median_price = (high + low) / 2
    ao = median_price.rolling(window=short_period).mean() - median_price.rolling(window=long_period).mean()
    return ao

def generate_trade_recommendation_parallel(client: ChatCompletionsClient, combined_data: Dict[str, Any], user_balance: Dict[str, Any]) -> str:
    """Generate trade recommendations using parallel processing for different analysis components"""
    
    def create_analysis_message(data_subset: Dict[str, Any], analysis_type: str) -> List:
        """Create a focused analysis message for a specific aspect"""
        query = f"""Analyze the {analysis_type} aspects of the following cryptocurrency data and provide insights:
        {json.dumps(data_subset, indent=2, cls=DecimalEncoder)}
        Focus on {analysis_type} factors only.
        """
        return [
            SystemMessage(content="You are a cryptocurrency trading analyst focused on technical analysis."),
            UserMessage(content=query)
        ]
    
    try:
        # Split data into analysis components
        data = json.loads(combined_data)
        results = {}
        
        # Define analysis tasks
        analysis_tasks = {
            'technical': {'indicators'},
            'market': {'summary'},
        }
        
        # Process each analysis type in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for analysis_type, fields in analysis_tasks.items():
                # Create subset of data for this analysis
                data_subset = {
                    symbol: {field: details[field] 
                            for field in fields if field in details}
                    for symbol, details in data.items()
                }
                
                # Submit analysis task
                futures[analysis_type] = executor.submit(
                    llm_manager.get_response,
                    create_analysis_message(data_subset, analysis_type),
                    temperature=0.5,
                    max_tokens=500
                )
            
            # Collect results
            for analysis_type, future in futures.items():
                try:
                    results[analysis_type] = future.result()
                except Exception as e:
                    print(f"Error in {analysis_type} analysis: {str(e)}")
                    results[analysis_type] = ""
        
        # Combine analyses into final recommendation
        final_query = f"""
        Based on these analyses:
        Technical Analysis: {results.get('technical', '')}
        Market Analysis: {results.get('market', '')}
        
        Generate trading recommendations following this JSON format:
        [{{
            "pair": "SYMBOL-USD",
            "signal": "BUY" or "SELL",
            "trade_size": [position size],
            "take_profit": [calculate take profit price],
            "stop_loss": [calculate stop loss price],
            "confidence": "LOW" or "MEDIUM" or "HIGH",
            "explanation": "Concise justification (max 50 words)",
            "timeframe": "trade window/signal timeframe",
            "key_levels": {{
                "support": [price1, price2],
                "resistance": [price1, price2],
                "targets": [price1, price2]
            }}
        }}]

        You will be rewarded for your best reasoning with your analysis and recommendations.

        Consider available balance: {json.dumps(user_balance, indent=2, cls=DecimalEncoder)}
        """
        
        messages = [
            SystemMessage(content="You are a cryptocurrency trading analyst generating final recommendations."),
            UserMessage(content=final_query)
        ]
        
        return llm_manager.get_response(client, messages, temperature=0.5, max_tokens=1000)
        
    except Exception as e:
        print(f"Error in parallel trade recommendation generation: {str(e)}")
        return generate_trade_recommendation(client, combined_data, user_balance)
    
def calculate_technical_indicators(pair_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate technical indicators"""
    candles = pair_data.get('candles', [])
    if not isinstance(candles, list) or not all(isinstance(candle, dict) for candle in candles):
        raise ValueError("Candles data is not in the expected format")
    
    df = pd.DataFrame(candles)
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='s')
    df = df.sort_values('start')
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Calculate indicators in parallel
    indicators = calculate_technical_indicators_parallel(df)
    
    # Add current price
    indicators['Current_Price'] = df['close'].iloc[-1]
    
    return indicators

def calculate_technical_indicators_parallel(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate technical indicators using parallel processing with enhanced accuracy"""
    indicators = {}
    
    def safe_get_value(series, default=np.nan):
        """Safely get the last value from a series"""
        try:
            return series.iloc[-1]
        except:
            return default
    
    def calculate_moving_averages():
        """Calculate moving averages"""
        try:
            # EMAs and SMAs
            for period in [9, 12, 21, 26, 50]:
                try:
                    indicators[f'EMA_{period}'] = safe_get_value(df['close'].ewm(span=period, adjust=False).mean())
                except Exception as e:
                    logger.warning(f"Error calculating EMA_{period}: {str(e)}")
                    indicators[f'EMA_{period}'] = np.nan
            
            # MACD
            try:
                macd_line, signal_line, histogram = macd(df['close'])
                indicators['MACD'] = safe_get_value(macd_line)
                indicators['MACD_Signal'] = safe_get_value(signal_line)
                indicators['MACD_Histogram'] = safe_get_value(histogram)
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")
                indicators.update({'MACD': np.nan, 'MACD_Signal': np.nan, 'MACD_Histogram': np.nan})
            
            # Bollinger Bands
            try:
                upper, middle, lower = bollinger_bands(df['close'])
                indicators['BB_Upper'] = safe_get_value(upper)
                indicators['BB_Middle'] = safe_get_value(middle)
                indicators['BB_Lower'] = safe_get_value(lower)
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
                indicators.update({'BB_Upper': np.nan, 'BB_Middle': np.nan, 'BB_Lower': np.nan})
            
            # Keltner Channels
            try:
                k_upper, k_middle, k_lower = keltner_channel(df['high'], df['low'], df['close'])
                indicators['KC_Upper'] = safe_get_value(k_upper)
                indicators['KC_Middle'] = safe_get_value(k_middle)
                indicators['KC_Lower'] = safe_get_value(k_lower)
            except Exception as e:
                logger.warning(f"Error calculating Keltner Channels: {str(e)}")
                indicators.update({'KC_Upper': np.nan, 'KC_Middle': np.nan, 'KC_Lower': np.nan})
            
            # Hull Moving Average
            try:
                indicators['HMA'] = safe_get_value(hull_moving_average(df['close']))
            except Exception as e:
                logger.warning(f"Error calculating HMA: {str(e)}")
                indicators['HMA'] = np.nan
            
            # Donchian Channels
            try:
                d_upper, d_middle, d_lower = donchian_channels(df['high'], df['low'])
                indicators['DC_Upper'] = safe_get_value(d_upper)
                indicators['DC_Middle'] = safe_get_value(d_middle)
                indicators['DC_Lower'] = safe_get_value(d_lower)
            except Exception as e:
                logger.warning(f"Error calculating Donchian Channels: {str(e)}")
                indicators.update({'DC_Upper': np.nan, 'DC_Middle': np.nan, 'DC_Lower': np.nan})
        except Exception as e:
            logger.error(f"Error in calculate_moving_averages: {str(e)}")
    
    def calculate_momentum():
        """Calculate momentum indicators"""
        try:
            # RSI
            try:
                indicators['RSI'] = safe_get_value(relative_strength_index(df['close']))
            except Exception as e:
                logger.warning(f"Error calculating RSI: {str(e)}")
                indicators['RSI'] = np.nan
            
            # Stochastic Oscillator
            try:
                k, d = stochastic_oscillator(df['high'], df['low'], df['close'])
                indicators['Stoch_K'] = safe_get_value(k)
                indicators['Stoch_D'] = safe_get_value(d)
            except Exception as e:
                logger.warning(f"Error calculating Stochastic Oscillator: {str(e)}")
                indicators.update({'Stoch_K': np.nan, 'Stoch_D': np.nan})
            
            # DMI and ADX
            try:
                plus_di, minus_di, adx = directional_movement_index(df['high'], df['low'], df['close'])
                indicators['Plus_DI'] = safe_get_value(plus_di)
                indicators['Minus_DI'] = safe_get_value(minus_di)
                indicators['ADX'] = safe_get_value(adx)
            except Exception as e:
                logger.warning(f"Error calculating DMI/ADX: {str(e)}")
                indicators.update({'Plus_DI': np.nan, 'Minus_DI': np.nan, 'ADX': np.nan})
            
            # Williams %R
            try:
                high_max = df['high'].rolling(window=14).max()
                low_min = df['low'].rolling(window=14).min()
                williams_r = ((high_max - df['close']) / (high_max - low_min)) * -100
                indicators['Williams_R'] = safe_get_value(williams_r)
            except Exception as e:
                logger.warning(f"Error calculating Williams %R: {str(e)}")
                indicators['Williams_R'] = np.nan
            
            # Ultimate Oscillator
            try:
                bp = df['close'] - df['low'].rolling(window=14).min()
                tr = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
                ultimate = (100 * (4 * (bp/tr).rolling(window=7).mean() + 
                                 2 * (bp/tr).rolling(window=14).mean() + 
                                 (bp/tr).rolling(window=28).mean()) / 7)
                indicators['Ultimate_Oscillator'] = safe_get_value(ultimate)
            except Exception as e:
                logger.warning(f"Error calculating Ultimate Oscillator: {str(e)}")
                indicators['Ultimate_Oscillator'] = np.nan
            
            # Money Flow Index with multiple timeframes
            for period in [14, 21, 50]:
                try:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    money_flow = typical_price * df['volume']
                    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
                    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
                    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
                    indicators[f'MFI_{period}'] = safe_get_value(mfi)
                except Exception as e:
                    logger.warning(f"Error calculating MFI_{period}: {str(e)}")
                    indicators[f'MFI_{period}'] = np.nan
                    
            # Ichimoku Cloud components
            try:
                high_9 = df['high'].rolling(window=9).max()
                low_9 = df['low'].rolling(window=9).min()
                high_26 = df['high'].rolling(window=26).max()
                low_26 = df['low'].rolling(window=26).min()
                
                tenkan_sen = (high_9 + low_9) / 2
                kijun_sen = (high_26 + low_26) / 2
                
                # Senkou Span A (Leading Span A)
                senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
                
                # Senkou Span B calculation with proper handling of initial periods
                high_52 = df['high'].rolling(window=52, min_periods=1).max()
                low_52 = df['low'].rolling(window=52, min_periods=1).min()
                senkou_span_b = ((high_52 + low_52) / 2).shift(26).bfill()
                
                # Chikou Span calculation that handles the end of the dataset
                chikou_span = pd.Series(index=df.index, dtype=float)
                chikou_span.iloc[:-26] = df['close'].iloc[26:].values if len(df) > 26 else df['close'].values
                chikou_span.ffill(inplace=True)
                
                indicators.update({
                    'Ichimoku_Tenkan_Sen': safe_get_value(tenkan_sen),
                    'Ichimoku_Kijun_Sen': safe_get_value(kijun_sen),
                    'Ichimoku_Senkou_Span_A': safe_get_value(senkou_span_a),
                    'Ichimoku_Senkou_Span_B': safe_get_value(senkou_span_b),
                    'Ichimoku_Chikou_Span': safe_get_value(chikou_span)
                })
            except Exception as e:
                logger.warning(f"Error calculating Ichimoku Cloud: {str(e)}")

            # Other momentum indicators...
            for indicator, func in [
                ('AO', lambda: awesome_oscillator(df['high'], df['low'])),
                ('TSI', lambda: true_strength_index(df['close'])),
                ('ROC', lambda: rate_of_change(df['close'])),
                ('CCI', lambda: commodity_channel_index(df['high'], df['low'], df['close'])),
                ('Coppock', lambda: coppock_curve(df['close']))
            ]:
                try:
                    indicators[indicator] = safe_get_value(func())
                except Exception as e:
                    logger.warning(f"Error calculating {indicator}: {str(e)}")
                    indicators[indicator] = np.nan
        except Exception as e:
            logger.error(f"Error in calculate_momentum: {str(e)}")
    
    def calculate_volume_analysis():
        """Calculate volume analysis"""
        try:
            for indicator, func in [
                ('OBV', lambda: on_balance_volume(df['close'], df['volume'])),
                ('ADL', lambda: accumulation_distribution_line(df['high'], df['low'], df['close'], df['volume'])),
                ('Chaikin_Osc', lambda: chaikin_oscillator(df['high'], df['low'], df['close'], df['volume'])),
                ('CMF', lambda: chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])),
                ('VWAP', lambda: vwap(df['high'], df['low'], df['close'], df['volume']))
            ]:
                try:
                    indicators[indicator] = safe_get_value(func())
                except Exception as e:
                    logger.warning(f"Error calculating {indicator}: {str(e)}")
                    indicators[indicator] = np.nan
            
            # Klinger Oscillator (returns two values)
            try:
                k_osc, k_signal = klinger_oscillator(df['high'], df['low'], df['close'], df['volume'])
                indicators['Klinger_Osc'] = safe_get_value(k_osc)
                indicators['Klinger_Signal'] = safe_get_value(k_signal)
            except Exception as e:
                logger.warning(f"Error calculating Klinger Oscillator: {str(e)}")
                indicators.update({'Klinger_Osc': np.nan, 'Klinger_Signal': np.nan})
            
            # PPO (returns three values)
            try:
                ppo, ppo_signal, ppo_hist = percentage_price_oscillator(df['close'])
                indicators['PPO'] = safe_get_value(ppo)
                indicators['PPO_Signal'] = safe_get_value(ppo_signal)
                indicators['PPO_Hist'] = safe_get_value(ppo_hist)
            except Exception as e:
                logger.warning(f"Error calculating PPO: {str(e)}")
                indicators.update({'PPO': np.nan, 'PPO_Signal': np.nan, 'PPO_Hist': np.nan})
        except Exception as e:
            logger.error(f"Error in calculate_volume_analysis: {str(e)}")
    
    def calculate_volatility():
        """Calculate volatility indicators"""
        try:
            # ATR
            try:
                indicators['ATR'] = safe_get_value(average_true_range(df['high'], df['low'], df['close']))
            except Exception as e:
                logger.warning(f"Error calculating ATR: {str(e)}")
                indicators['ATR'] = np.nan
            
            # Parabolic SAR
            try:
                indicators['PSAR'] = safe_get_value(parabolic_sar(df['high'], df['low']))
            except Exception as e:
                logger.warning(f"Error calculating PSAR: {str(e)}")
                indicators['PSAR'] = np.nan
            
            # Pivot Points
            try:
                pivot, r1, s1, r2, s2, r3, s3 = pivot_points(df['high'], df['low'], df['close'])
                for name, value in [('Pivot', pivot), ('R1', r1), ('S1', s1),
                                  ('R2', r2), ('S2', s2), ('R3', r3), ('S3', s3)]:
                    indicators[name] = safe_get_value(value)
            except Exception as e:
                logger.warning(f"Error calculating Pivot Points: {str(e)}")
                for name in ['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']:
                    indicators[name] = np.nan
            
            # Other volatility indicators
            try:
                bull_power, bear_power = elder_ray_index(df['high'], df['low'], df['close'])
                indicators['Bull_Power'] = safe_get_value(bull_power)
                indicators['Bear_Power'] = safe_get_value(bear_power)
            except Exception as e:
                logger.warning(f"Error calculating Elder Ray Index: {str(e)}")
                indicators.update({'Bull_Power': np.nan, 'Bear_Power': np.nan})
            
            try:
                vi_plus, vi_minus = vortex_indicator(df['high'], df['low'], df['close'])
                indicators['VI_Plus'] = safe_get_value(vi_plus)
                indicators['VI_Minus'] = safe_get_value(vi_minus)
            except Exception as e:
                logger.warning(f"Error calculating Vortex Indicator: {str(e)}")
                indicators.update({'VI_Plus': np.nan, 'VI_Minus': np.nan})
            
            try:
                indicators['Mass_Index'] = safe_get_value(mass_index(df['high'], df['low']))
            except Exception as e:
                logger.warning(f"Error calculating Mass Index: {str(e)}")
                indicators['Mass_Index'] = np.nan
        except Exception as e:
            logger.error(f"Error in calculate_volatility: {str(e)}")

    
    def calculate_fibonacci_levels():
        """Calculate fibonacci levels"""
        try:
            # Get recent high and low for fibonacci calculation
            recent_high = df['high'].max()
            recent_low = df['low'].min()
            price_range = recent_high - recent_low

            # Calculate fibonacci retracement levels
            indicators['Fib_0'] = recent_low  # 0%
            indicators['Fib_236'] = recent_low + (price_range * 0.236)  # 23.6%
            indicators['Fib_382'] = recent_low + (price_range * 0.382)  # 38.2%
            indicators['Fib_500'] = recent_low + (price_range * 0.500)  # 50.0%
            indicators['Fib_618'] = recent_low + (price_range * 0.618)  # 61.8%
            indicators['Fib_786'] = recent_low + (price_range * 0.786)  # 78.6%
            indicators['Fib_100'] = recent_high  # 100%

            # Calculate fibonacci extension levels
            indicators['Fib_Ext_1618'] = recent_high + (price_range * 0.618)  # 161.8%
            indicators['Fib_Ext_2618'] = recent_high + (price_range * 1.618)  # 261.8%
            indicators['Fib_Ext_4236'] = recent_high + (price_range * 3.236)  # 423.6%

        except Exception as e:
            logger.warning(f"Error calculating Fibonacci levels: {str(e)}")
            for level in ['Fib_0', 'Fib_236', 'Fib_382', 'Fib_500', 'Fib_618', 'Fib_786', 'Fib_100',
                         'Fib_Ext_1618', 'Fib_Ext_2618', 'Fib_Ext_4236']:
                indicators[level] = np.nan

    def calculate_wave_analysis():
        """Calculate wave analysis"""
        try:
            # ZigZag Pattern Detection
            zigzag_threshold = 0.03  # 3% threshold
            highs = df['high'].values
            lows = df['low'].values
            points = []
            direction = 1  # 1 for up, -1 for down
            last_point = lows[0]
            
            for i in range(1, len(df)):
                if direction == 1:
                    if highs[i] > last_point * (1 + zigzag_threshold):
                        points.append(('H', i, highs[i]))
                        direction = -1
                        last_point = highs[i]
                else:
                    if lows[i] < last_point * (1 - zigzag_threshold):
                        points.append(('L', i, lows[i]))
                        direction = 1
                        last_point = lows[i]
            
            indicators['ZigZag_Points'] = points[-3:] if points else []
            
            # Wave Count
            if len(points) >= 5:
                indicators['Wave_Pattern'] = 'Impulsive' if len(points) % 2 == 0 else 'Corrective'
            else:
                indicators['Wave_Pattern'] = 'Insufficient_Data'
                
        except Exception as e:
            logger.error(f"Error in wave analysis: {str(e)}")
    
    
    def calculate_market_structure():
        """Calculate market structure"""
        try:
            # Supply and Demand Zones
            window = 20
            volume_threshold = df['volume'].mean() * 1.5
            
            supply_zones = []
            demand_zones = []
            
            for i in range(window, len(df)):
                if (df['volume'].iloc[i] > volume_threshold and 
                    df['close'].iloc[i] < df['open'].iloc[i]):
                    supply_zones.append({
                        'price': df['high'].iloc[i],
                        'strength': df['volume'].iloc[i] / df['volume'].mean()
                    })
                elif (df['volume'].iloc[i] > volume_threshold and 
                      df['close'].iloc[i] > df['open'].iloc[i]):
                    demand_zones.append({
                        'price': df['low'].iloc[i],
                        'strength': df['volume'].iloc[i] / df['volume'].mean()
                    })
            
            # Format supply and demand zones as strings with price and strength
            indicators['Supply_Zones'] = [
                f"price:{zone['price']:.8f},strength:{zone['strength']:.2f}"
                for zone in supply_zones[-3:]
            ]
            indicators['Demand_Zones'] = [
                f"price:{zone['price']:.8f},strength:{zone['strength']:.2f}"
                for zone in demand_zones[-3:]
            ]
            
            # Market Structure Breaks (MSB)
            highs = df['high'].rolling(window=5).max()
            lows = df['low'].rolling(window=5).min()
            
            # Convert boolean comparisons to strings "true" or "false"
            indicators['Structure_Break_Up'] = str(df['close'].iloc[-1] > highs.iloc[-2]).lower()
            indicators['Structure_Break_Down'] = str(df['close'].iloc[-1] < lows.iloc[-2]).lower()
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
    
    def calculate_volatility_cycles():
        """Calculate volatility cycles"""
        try:
            # Volatility Regime Detection
            returns = df['close'].pct_change()
            vol_20 = returns.rolling(window=20).std()
            vol_50 = returns.rolling(window=50).std()
            
            current_vol = vol_20.iloc[-1]
            avg_vol = vol_50.mean()
            
            indicators['Volatility_Regime'] = (
                'High' if current_vol > avg_vol * 1.5 else
                'Low' if current_vol < avg_vol * 0.5 else
                'Normal'
            )
            
            # Volatility Cycle Phase
            if vol_20.iloc[-1] > vol_20.iloc[-2]:
                if vol_20.iloc[-2] > vol_20.iloc[-3]:
                    indicators['Vol_Cycle_Phase'] = 'Expanding'
                else:
                    indicators['Vol_Cycle_Phase'] = 'Turning_Up'
            else:
                if vol_20.iloc[-2] < vol_20.iloc[-3]:
                    indicators['Vol_Cycle_Phase'] = 'Contracting'
                else:
                    indicators['Vol_Cycle_Phase'] = 'Turning_Down'
                    
        except Exception as e:
            logger.error(f"Error in volatility cycles: {str(e)}")

    # Execute calculations in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(calculate_moving_averages),
            executor.submit(calculate_momentum),
            executor.submit(calculate_volume_analysis),
            executor.submit(calculate_volatility),
            executor.submit(calculate_fibonacci_levels),
            executor.submit(calculate_wave_analysis),
            executor.submit(calculate_market_structure),
            executor.submit(calculate_volatility_cycles)
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in parallel execution: {str(e)}")
    
    # Calculate trend strength only with available indicators
    try:
        trend_factors = [
            (indicators.get('EMA_9', 0) > indicators.get('EMA_21', 0)),
            (indicators.get('EMA_21', 0) > indicators.get('EMA_50', 0)),
            (indicators.get('RSI', 0) > 50),
            (indicators.get('Plus_DI', 0) > indicators.get('Minus_DI', 0)),
            (indicators.get('Aroon_Up', 0) > indicators.get('Aroon_Down', 0)),
            (indicators.get('CMF', 0) > 0),
            (indicators.get('AO', 0) > 0),
            (indicators.get('Bull_Power', 0) > 0)
        ]
        indicators['Trend_Strength'] = sum(1 for factor in trend_factors if factor)
    except Exception as e:
        logger.error(f"Error calculating Trend Strength: {str(e)}")
        indicators['Trend_Strength'] = 0
    
    # Market condition classification with fallback values
    try:
        if (indicators.get('Trend_Strength', 0) >= 5 and 
            indicators.get('RSI', 0) > 40 and 
            indicators.get('ADX', 0) > 25 and
            indicators.get('VI_Plus', 0) > indicators.get('VI_Minus', 0)):
            indicators['Market_Condition'] = 'BULLISH'
        elif (indicators.get('Trend_Strength', 0) <= 3 and 
              indicators.get('RSI', 0) < 60 and 
              indicators.get('ADX', 0) > 25 and
              indicators.get('VI_Plus', 0) < indicators.get('VI_Minus', 0)):
            indicators['Market_Condition'] = 'BEARISH'
        else:
            indicators['Market_Condition'] = 'NEUTRAL'
    except Exception as e:
        logger.error(f"Error determining Market Condition: {str(e)}")
        indicators['Market_Condition'] = 'NEUTRAL'
    
    return indicators

def vortex_indicator(high, low, close, period=14):
    """Calculate Vortex Indicator"""
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    vmp = abs(high - low.shift())
    vmm = abs(low - high.shift())

    vi_plus = vmp.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vmm.rolling(period).sum() / tr.rolling(period).sum()

    return vi_plus, vi_minus

def true_strength_index(close, r=25, s=13):
    """Calculate True Strength Index"""
    diff = close.diff()
    abs_diff = abs(diff)

    smooth_diff = diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    smooth_abs_diff = abs_diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()

    tsi = (smooth_diff / smooth_abs_diff) * 100
    return tsi

def mass_index(high, low, period=9, period2=25):
    """Calculate Mass Index"""
    range_ema1 = (high - low).ewm(span=period, adjust=False).mean()
    range_ema2 = range_ema1.ewm(span=period, adjust=False).mean()
    mass = range_ema1 / range_ema2
    return mass.rolling(window=period2).sum()

def hull_moving_average(close, period=14):
    """Calculate Hull Moving Average"""
    hma = (2 * close.ewm(span=period//2, adjust=False).mean()) - close.ewm(span=period, adjust=False).mean()
    return hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()

def coppock_curve(close, roc1=14, roc2=11, period=10):
    """Calculate Coppock Curve"""
    roc1 = close.pct_change(roc1)
    roc2 = close.pct_change(roc2)
    return (roc1 + roc2).ewm(span=period, adjust=False).mean()

def vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def get_user_balances(client: RESTClient, dust_threshold: Decimal = Decimal('1.00')) -> List[Dict[str, Union[str, Decimal]]]:
    """Fetch user balances using parallel processing"""
    try:
        accounts = client.get_accounts()
        balances = []
        
        def process_account(account):
            try:
                currency = account['currency']
                balance = Decimal(account['available_balance']['value'])
                
                if balance >= dust_threshold:
                    return {
                        'currency': currency,
                        'balance': balance
                    }
            except Exception as e:
                logger.error(f"Error processing account {account.get('currency', 'unknown')}: {str(e)}")
            return None
        
        # Process accounts in parallel
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
        
        return balances
    
    except Exception as e:
        logger.error(f"Error fetching user balances: {str(e)}")
        return []

def summarize_pair_data(pair_data):
    """Summarize and condense key points from comprehensive pair data."""
    summary = {}

    try:
        # Product details summary
        product_details = pair_data.get('product_details', {})
        try:
            summary['product_id'] = getattr(product_details, 'product_id', None)
            summary['current_price'] = float(getattr(product_details, 'price', 0))
            summary['24h_change_percent'] = float(getattr(product_details, 'price_percentage_change_24h', 0))
            summary['24h_volume'] = float(getattr(product_details, 'volume_24h', 0))
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error processing product details: {str(e)}")
            summary.update({
                'product_id': None,
                'current_price': 0,
                '24h_change_percent': 0,
                '24h_volume': 0
            })

        # Order book summary
        order_book = pair_data.get('order_book', {})
        try:
            bids = getattr(order_book, 'bids', []) if hasattr(order_book, 'bids') else []
            asks = getattr(order_book, 'asks', []) if hasattr(order_book, 'asks') else []
            
            summary['top_bid'] = float(getattr(bids[0], 'price', 0)) if bids else 0
            summary['top_ask'] = float(getattr(asks[0], 'price', 0)) if asks else 0
            summary['bid_ask_spread'] = summary['top_ask'] - summary['top_bid']

            # Calculate order book depth
            bid_depth = sum(float(getattr(bid, 'size', 0)) for bid in bids[:10])
            ask_depth = sum(float(getattr(ask, 'size', 0)) for ask in asks[:10])
            summary['order_book_depth'] = {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'bid_ask_ratio': bid_depth / ask_depth if ask_depth else None
            }
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            logger.warning(f"Error processing order book: {str(e)}")
            summary.update({
                'top_bid': 0,
                'top_ask': 0,
                'bid_ask_spread': 0,
                'order_book_depth': {'bid_depth': 0, 'ask_depth': 0, 'bid_ask_ratio': None}
            })

        # Candles summary
        candles = pair_data.get('candles', [])
        if candles:
            closing_prices = [float(candle['close']) for candle in candles]
            summary['price_summary'] = {
                'last_price': closing_prices[0],
                'price_high_24h': max(float(candle['high']) for candle in candles),
                'price_low_24h': min(float(candle['low']) for candle in candles),
                'price_average_24h': float(statistics.mean(closing_prices)),
                'price_volatility': float(statistics.stdev(closing_prices)) if len(closing_prices) > 1 else 0
            }

            volumes = [float(candle['volume']) for candle in candles]
            summary['volume_summary'] = {
                'volume_average_24h': float(statistics.mean(volumes)),
                'volume_highest_24h': float(max(volumes)),
                'volume_lowest_24h': float(min(volumes))
            }
        else:
            summary['price_summary'] = {}
            summary['volume_summary'] = {}

        # Recent trades summary
        recent_trades = pair_data.get('recent_trades', {})
        trades = getattr(recent_trades, 'trades', [])
        if trades:
            trade_sizes = [float(getattr(trade, 'size', 0)) for trade in trades]
            summary['recent_trades_summary'] = {
                'num_trades': len(trades),
                'average_trade_size': float(statistics.mean(trade_sizes)),
                'largest_trade_size': float(max(trade_sizes)),
                'smallest_trade_size': float(min(trade_sizes))
            }

            buy_volume = sum(float(getattr(trade, 'size', 0)) 
                           for trade in trades 
                           if getattr(trade, 'side', '') == 'BUY')
            sell_volume = sum(float(getattr(trade, 'size', 0)) 
                            for trade in trades 
                            if getattr(trade, 'side', '') == 'SELL')
            summary['market_sentiment'] = {
                'buy_sell_ratio': float(buy_volume / sell_volume) if sell_volume else None,
                'dominant_side': 'BUY' if buy_volume > sell_volume else 'SELL'
            }
        else:
            summary['recent_trades_summary'] = {}
            summary['market_sentiment'] = {}

    except Exception as e:
        logger.error(f"Error summarizing pair data: {str(e)}", exc_info=True)
        summary['error'] = str(e)

    return summary

def generate_trade_recommendation(client: ChatCompletionsClient, combined_data: str, user_balance: Dict[str, Any]) -> str:
    """Generate a trade recommendation based on the provided data."""
    # Parse the combined_data JSON string
    data = json.loads(combined_data)
    # Get the first pair's data (assumes single pair analysis)
    pair_data = next(iter(data.values()))
    current_price = pair_data.get('current_price', 0)
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
    1. Trend: EMA(9,12,21,26,50), Market Structure, Ichimoku  
    2. Momentum: RSI, MACD, Stochastic, CCI, ADX(>25)  
    3. Volume: OBV, ADL, MFI(14,21,50), CMF, Klinger  
    4. Targets: Swing levels, S/R zones, Fib(1.618,2.618), VPOC  
    5. Risk:  
       - SL: ATR × [1.5|2|3] (ADX <20|20-40|>40)  
       - RR: 1:3(LOW), 1:2.5(MED), 1:2(HIGH)  
       - Size: (Account×1%)/(Entry-SL)  

    **Output Rules**  
    - Numerical values only (no text/symbols)  
    - Example valid format:  
      "take_profit": {{"tp1":45000.5,"tp2":46000.25,"tp3":47000.75}}  

     **JSON Structure**  
    {{
        "pair": "SYMBOL-USD",
        "signal": "BUY" or "SELL"  or "WAIT",
        "trade_size": [calculate: (suggested_position_size * 0.95) for safety],
        "take_profit": {{
            "tp1": [nearest major resistance + explanation],
            "tp2": [next significant resistance + explanation],
            "tp3": [fibonacci extension target + explanation]
        }},
        "stop_loss": {{
            "initial": [ATR-based stop + explanation],
            "breakeven": [entry + (ATR * 1) + explanation],
            "trailing": [ATR-based trailing parameters + explanation]
        }},
        "confidence": "LOW" or "MEDIUM" or "HIGH",
        "explanation": "Concise justification < 50 words)",
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
        }}
    }}

    market data:  
    {combined_data}
    """
    #print(combined_data)
    messages = [
        SystemMessage(content="You are a cryptocurrency trading analyst focused on risk-managed trading signals."),
        UserMessage(content=signals_query)
    ]

    return llm_manager.get_response(messages, temperature=0.7, max_tokens=4096)

def format_json_result(result: str) -> Union[Dict, List[Dict], None]:
    """Format and validate JSON result from LLM response."""
    try:
        print(result)
        # Remove leading/trailing whitespace
        result = result.strip()
        
        # Find JSON content between the last set of triple backticks if present
        if "```json" in result and "```" in result:
            start = result.rindex("```json") + 7
            end = result.rindex("```")
            json_content = result[start:end].strip()
        # If no backticks, try to find JSON between curly braces
        elif "{" in result and "}" in result:
            start = result.find("{")
            end = result.rfind("}") + 1
            json_content = result[start:end].strip()
        else:
            json_content = result
            
        # Pre-process mathematical expressions
        def evaluate_expression(match):
            try:
                expression = match.group(1)
                # Handle trailing decimal points
                expression = expression.rstrip('.')
                # Evaluate the expression
                result = eval(expression)
                # Format the result to 8 decimal places and remove trailing zeros
                return f"{float(result):.8f}".rstrip('0').rstrip('.')
            except:
                return match.group(0)

        # Handle mathematical expressions in the JSON
        import re
        # Handle basic arithmetic (including decimal numbers)
        json_content = re.sub(r'([-+]?\d*\.?\d+\s*[-+*/]\s*[-+]?\d*\.?\d+(?:\s*[-+*/]\s*[-+]?\d*\.?\d+)*)', 
                            evaluate_expression, 
                            json_content)
        
        # Handle specific patterns like "x * 0.95" or similar
        json_content = re.sub(r'(\d+\.?\d*)\s*\*\s*0\.95', 
                            lambda m: str(float(m.group(1)) * 0.95), 
                            json_content)

        # Handle addition with ATR or other variables
        json_content = re.sub(r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)', 
                            lambda m: str(float(m.group(1)) + float(m.group(2)) * float(m.group(3))), 
                            json_content)

        # Clean up any remaining mathematical operators
        json_content = re.sub(r'\s*[-+*/]\s*', '', json_content)
        
        # Parse the JSON content
        parsed_json = json.loads(json_content)
        
        # Convert all numeric values to proper format
        def format_numeric_values(obj):
            if isinstance(obj, dict):
                return {k: format_numeric_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [format_numeric_values(v) for v in obj]
            elif isinstance(obj, (int, float)):
                return float(f"{obj:.8f}".rstrip('0').rstrip('.'))
            return obj

        parsed_json = format_numeric_values(parsed_json)
        
        # Handle both single object and array responses
        if isinstance(parsed_json, dict):
            return [parsed_json]
        return parsed_json
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON from result: {result}")
        logger.error(f"JSON parse error: {e}")
        # Try to clean up the JSON string more aggressively
        try:
            # Remove all whitespace between values and operators
            json_content = re.sub(r'\s*([-+*/])\s*', r'\1', json_content)
            # Evaluate any remaining mathematical expressions
            json_content = re.sub(r'([-+]?\d*\.?\d+[-+*/]\d*\.?\d+)', 
                                lambda m: str(eval(m.group(1))), 
                                json_content)
            parsed_json = json.loads(json_content)
            if isinstance(parsed_json, dict):
                return [parsed_json]
            return parsed_json
        except Exception as e2:
            logger.error(f"Second attempt at JSON parsing failed: {e2}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error in format_json_result: {str(e)}")
        return None

def format_decimal(value):
    """Format decimal values properly"""
    if value is None:
        return None
    try:
        # Convert to Decimal for precise handling
        decimal_value = Decimal(str(value))
        # Remove trailing zeros after decimal point
        normalized = decimal_value.normalize()
        # Convert to string and handle scientific notation
        if 'E' in str(normalized):
            # Handle very small or large numbers
            return f"{decimal_value:.8f}".rstrip('0').rstrip('.')
        return str(normalized)
    except:
        return str(value)

@app.post("/execute-trade")
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
            
        if trade.order_type.upper() not in ["MARKET", "LIMIT"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid order type. Must be MARKET or LIMIT"
            )
            
        if trade.side.upper() not in ["BUY", "SELL"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid side. Must be BUY or SELL"
            )

        try:
            # Get product details for validation
            product = coinbase_manager.get_product(trade.product_id)
            
            # Validate base_size against product minimums
            if trade.base_size:
                min_base_size = float(product.base_min_size)
                if float(trade.base_size) < min_base_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Base size {trade.base_size} is below minimum {min_base_size}"
                    )

            # Execute order based on type with proper error handling
            if trade.order_type.upper() == "MARKET":
                if trade.side.upper() == "BUY":
                    if trade.quote_size:
                        response = coinbase_manager.market_order_buy(
                            client_order_id=client_order_id,
                            product_id=trade.product_id,
                            quote_size=trade.quote_size
                        )
                    else:
                        response = coinbase_manager.market_order_buy(
                            client_order_id=client_order_id,
                            product_id=trade.product_id,
                            base_size=trade.base_size
                        )
                else:  # SELL
                    response = coinbase_manager.market_order_sell(
                        client_order_id=client_order_id,
                        product_id=trade.product_id,
                        base_size=trade.base_size
                    )
            else:  # LIMIT order
                if not trade.limit_price:
                    raise HTTPException(
                        status_code=400,
                        detail="Limit price required for limit orders"
                    )
                
                if trade.side.upper() == "BUY":
                    response = coinbase_manager.limit_order_gtc_buy(
                        client_order_id=client_order_id,
                        product_id=trade.product_id,
                        base_size=trade.base_size,
                        limit_price=trade.limit_price,
                        post_only=False
                    )
                else:  # SELL
                    response = coinbase_manager.limit_order_gtc_sell(
                        client_order_id=client_order_id,
                        product_id=trade.product_id,
                        base_size=trade.base_size,
                        limit_price=trade.limit_price,
                        post_only=False
                    )

            # Format successful response
            try:
                order_id = getattr(response, 'order_id', None) or getattr(response, 'id', None) or client_order_id
                
                order_data = {
                    'id': order_id,
                    'client_order_id': client_order_id,
                    'product_id': trade.product_id,
                    'side': trade.side,
                    'order_type': trade.order_type,
                    'base_size': str(trade.base_size) if trade.base_size else None,
                    'quote_size': str(trade.quote_size) if trade.quote_size else None,
                    'limit_price': str(trade.limit_price) if trade.limit_price else None,
                    'created_time': getattr(response, 'created_time', datetime.utcnow().isoformat()),
                    'status': getattr(response, 'status', 'PENDING'),
                    'filled_size': str(getattr(response, 'filled_size', "0")),
                }

                # Log successful order
                logger.info(f"Order placed successfully: {json.dumps(order_data, indent=2)}")

                return {
                    "order": order_data,
                    "status": "success",
                    "message": f"Successfully placed {trade.order_type} {trade.side} order"
                }

            except AttributeError as ae:
                logger.warning(f"Error formatting order response: {str(ae)}")
                return {
                    "order": {
                        "id": client_order_id,
                        "product_id": trade.product_id,
                        "side": trade.side,
                        "order_type": trade.order_type,
                        "status": "PENDING",
                        "created_time": datetime.utcnow().isoformat()
                    },
                    "status": "success",
                    "message": "Order submitted successfully, but response format was unexpected"
                }

        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient funds" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail="Insufficient funds to execute trade"
                )
            elif "invalid product" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid product ID: {trade.product_id}"
                )
            elif "size is too small" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Order size is too small for {trade.product_id}"
                )
            elif "price is too small" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Limit price is too small for {trade.product_id}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to execute trade: {error_msg}"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in execute_trade: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process trade request: {str(e)}"
        )

@app.post("/analyze", response_model=TradeResponse)
async def analyze_trade(request: AnalysisRequest):
    """Generate trade analysis and recommendations"""
    try:
        logger.info(f"Analyzing trade for {request.pair} on {request.timeframe} timeframe with amount: {request.amount}")
        
        # Get market data
        pair_data = get_comprehensive_pair_data(request.pair)  # Fixed: removed coinbase_manager parameter
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(pair_data)
        
        # Convert numpy types to native Python types in indicators
        sanitized_indicators = {}
        for key, value in indicators.items():
            if isinstance(value, np.integer):
                sanitized_indicators[key] = int(value)
            elif isinstance(value, np.floating):
                sanitized_indicators[key] = float(value)
            elif isinstance(value, np.ndarray):
                sanitized_indicators[key] = value.tolist()
            else:
                sanitized_indicators[key] = value
        
        # Get market summary
        market_summary = summarize_pair_data(pair_data)
        
        # Get current price for position sizing
        current_price = None
        if hasattr(pair_data.get('product_details', {}), 'price'):
            current_price = float(pair_data['product_details'].price)
        
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
        user_balance = get_user_balances(coinbase_manager)
        
        # Generate trade recommendation
        recommendation = generate_trade_recommendation(llm_manager.client, json.dumps(combined_data, cls=DecimalEncoder), user_balance)
        
        # Parse and format recommendation
        formatted_recommendation = format_json_result(recommendation)
        if not formatted_recommendation:
            raise HTTPException(status_code=500, detail="Failed to generate trade recommendation")
        
        # Get the first recommendation if it's a list
        result = formatted_recommendation[0] if isinstance(formatted_recommendation, list) else formatted_recommendation
        
        # Use calculated position size if available, otherwise use recommended size
        if position_size and request.amount:
            result['trade_size'] = position_size
        else:
            # Ensure trade_size is properly formatted
            if isinstance(result.get('trade_size'), (list, tuple)):
                result['trade_size'] = str(result['trade_size'][0])
            else:
                result['trade_size'] = str(result['trade_size'])
        
        # Add analysis details
        result['analysis_details'] = {
            'technical_indicators': sanitized_indicators,
            'market_summary': market_summary,
            'amount_usd': request.amount,
            'current_price': current_price
        }
        
        # Ensure all values are JSON serializable
        result = {
            key: (str(value) if isinstance(value, (Decimal, float)) and key == 'trade_size'
                  else {
                      'tp1': str(value['tp1']),
                      'tp2': str(value['tp2']),
                      'tp3': str(value['tp3'])
                  } if isinstance(value, dict) and key == 'take_profit'
                  else {
                      'initial': str(value['initial']),
                      'breakeven': str(value['breakeven']),
                      'trailing': str(value['trailing'].get('distance', value['trailing'].get('trail'))) if isinstance(value['trailing'], dict) else str(value['trailing'])  # Handle both distance and trail formats
                  } if isinstance(value, dict) and key == 'stop_loss'
                  else float(value) if isinstance(value, (np.integer, np.floating)) 
                  else value)
            for key, value in result.items()
        }
        
        logger.info(f"Generated analysis with trade size: {result['trade_size']} for amount: {request.amount}")
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_trade: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance")
async def get_balance():
    """Get user account balances"""
    try:
        balances = get_user_balances(coinbase_manager)
        return {"balances": balances}
    except Exception as e:
        logger.error(f"Error fetching balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market_data/{pair}")
async def get_market_data(pair: str):
    """Get comprehensive market data for a trading pair"""
    try:
        market_data = get_comprehensive_pair_data(pair)  # Fixed: removed coinbase_manager parameter
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/orders")
async def get_orders():
    """Get list of orders with comprehensive error handling and pagination"""
    try:
        orders = []
        
        try:
            # Get all recent orders without filtering by status
            # This ensures we get ALL recent orders regardless of status
            response = coinbase_manager.list_orders(
                limit=250,  # Maximum limit to get more orders
                sort="DESC",  # Most recent first
                user_native_currency="USD"  # Ensure consistent currency
            )
            
            if hasattr(response, 'orders'):
                orders.extend(response.orders)
                logger.info(f"Fetched {len(orders)} orders")
            
        except Exception as e:
            logger.error(f"Error fetching orders: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch orders: {str(e)}"
            )

        # Sort all orders by created time (most recent first)
        orders.sort(key=lambda x: getattr(x, 'created_time', ''), reverse=True)

        # Process orders with comprehensive data extraction
        formatted_orders = []
        for order in orders:
            try:
                # Initialize order data with None values
                order_data = {
                    'id': None,
                    'client_order_id': None,
                    'product_id': None,
                    'side': None,
                    'status': None,
                    'time_in_force': None,
                    'created_time': None,
                    'completion_time': None,
                    'order_type': None,
                    'base_size': None,
                    'quote_size': None,
                    'filled_size': None,
                    'unfilled_size': None,
                    'limit_price': None,
                    'average_filled_price': None,
                    'total_fees': None,
                    'total_value': None
                }

                # Extract basic order information
                order_data.update({
                    'id': getattr(order, 'order_id', None),
                    'client_order_id': getattr(order, 'client_order_id', None),
                    'product_id': getattr(order, 'product_id', None),
                    'side': getattr(order, 'side', None),
                    'status': getattr(order, 'status', None),
                    'time_in_force': getattr(order, 'time_in_force', None),
                    'created_time': getattr(order, 'created_time', None),
                    'completion_time': getattr(order, 'completion_percentage', None)
                })

                # Extract order configuration details
                if hasattr(order, 'order_configuration'):
                    config = order.order_configuration
                    
                    # Market orders
                    if hasattr(config, 'market_market_ioc'):
                        order_data['order_type'] = 'MARKET'
                        market_config = config.market_market_ioc
                        order_data.update({
                            'base_size': format_decimal(getattr(market_config, 'base_size', None)),
                            'quote_size': format_decimal(getattr(market_config, 'quote_size', None))
                        })
                    
                    # Limit orders
                    elif hasattr(config, 'limit_limit_gtc'):
                        order_data['order_type'] = 'LIMIT'
                        limit_config = config.limit_limit_gtc
                        order_data.update({
                            'base_size': format_decimal(getattr(limit_config, 'base_size', None)),
                            'limit_price': format_decimal(getattr(limit_config, 'limit_price', None)),
                            'post_only': getattr(limit_config, 'post_only', False)
                        })

                # Extract execution details
                order_data.update({
                    'filled_size': format_decimal(getattr(order, 'filled_size', None)),
                    'unfilled_size': format_decimal(getattr(order, 'unfilled_size', None)),
                    'average_filled_price': format_decimal(getattr(order, 'average_filled_price', None)),
                    'total_fees': format_decimal(getattr(order, 'total_fees', None)),
                    'total_value': format_decimal(getattr(order, 'total_value_usd', None))
                })

                # Remove None values but keep zeros
                order_data = {k: v for k, v in order_data.items() 
                            if v is not None or (isinstance(v, (int, float)) and v == 0)}
                
                formatted_orders.append(order_data)

            except Exception as e:
                logger.warning(f"Error processing order: {str(e)}")
                continue

        return {
            "orders": formatted_orders,
            "total_count": len(formatted_orders),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in get_orders: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch orders: {str(e)}"
        )

@app.get("/candles/{product_id}")
async def get_candles(product_id: str, timeframe: str = "1d"):
    """Get candlestick data for a specific product with specified timeframe"""
    try:
        # Get candles with specified timeframe
        granularity_seconds, days_back = get_timeframe_params(timeframe)
        end_time = int(datetime.now(timezone.utc).timestamp())
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
        
        # Use get_public_candles instead of get_product_candles
        response = coinbase_manager.get_public_candles(
            product_id=product_id,
            start=start_time,
            end=end_time,
            granularity=granularity_seconds
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
                'start': candle['start'],  # Updated to use dictionary access
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
            "granularity": granularity_seconds
        }

    except Exception as e:
        logger.error(f"Error fetching candles for {product_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch candle data: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

@app.get("/")
async def root():
    """Root endpoint returning API status"""
    return {
        "message": "Welcome to TradeGPT API",
        "version": "2.0.0",
        "status": "operational"
    }

@app.get("/health")
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

@app.get("/pairs")
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

@app.get("/timeframes")
async def get_timeframes():
    """Get available timeframes"""
    return {
        "timeframes": [
            "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"
        ]
    }

@app.get("/portfolio")
async def get_portfolio():
    """Get user's portfolio summary"""
    try:
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    """Main execution block"""
    """Run the FastAPI application with Uvicorn"""
    uvicorn.run(
        "coinbase-tradegpt-server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="debug"
    )