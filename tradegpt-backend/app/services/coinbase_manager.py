"""
Coinbase API Manager
Handles interactions with the Coinbase Advanced Trade API
"""
import os
from dotenv import load_dotenv
from coinbase.rest import RESTClient
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
        load_dotenv()  # Ensure environment variables are loaded
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