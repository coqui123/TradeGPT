"""
Pydantic models for the API
"""
from typing import Dict, Any, List, Union, Literal, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from decimal import Decimal

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