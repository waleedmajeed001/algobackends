from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import yfinance as yf
import asyncio
import json
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Binance client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

# Configure client to use server time
client.timestamp_offset = 0
client.timestamp = lambda: int(time.time() * 1000)

# Time synchronization
def get_server_time():
    try:
        server_time = client.get_server_time()
        logger.info(f"Server time: {server_time['serverTime']}")
        return server_time['serverTime']
    except Exception as e:
        logger.error(f"Error getting server time: {e}")
        return None

def sync_time():
    server_time = get_server_time()
    if server_time:
        local_time = int(time.time() * 1000)
        time_diff = server_time - local_time
        logger.info(f"Time difference: {time_diff}ms")
        return time_diff
    return 0

# Initial time sync
time_offset = sync_time()
logger.info(f"Initial time offset: {time_offset}ms")

# Cache for symbol info
symbol_info_cache = {}

def get_symbol_info(symbol):
    if symbol not in symbol_info_cache:
        symbol_info_cache[symbol] = client.get_symbol_info(symbol)
    return symbol_info_cache[symbol]

# Global variables
SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 60
AUTO_TRADE_ENABLED = False

class TradeRequest(BaseModel):
    amount: float = None  # Optional amount in USDT
    auto_trade: bool = False

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

async def get_market_data():
    try:
        # Get historical klines/candlestick data with retry mechanism
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get historical klines/candlestick data
                klines = client.get_klines(
                    symbol=SYMBOL,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=100
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert price columns to float
                df['close'] = df['close'].astype(float)
                
                # Calculate RSI
                rsi = calculate_rsi(df['close'], RSI_PERIOD)
                current_rsi = rsi.iloc[-1]
                
                return current_rsi, df['close'].iloc[-1]
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Error getting market data, retrying... (Attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(1)  # Wait a bit before retrying
                    continue
                else:
                    raise e
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return None, None

async def execute_buy(amount: float = None):
    try:
        # Get current price
        current_price = float(client.get_symbol_ticker(symbol=SYMBOL)['price'])
        logger.info(f"Current price: {current_price}")
        
        if amount is None:
            # If no amount specified, use 95% of available USDT balance
            balance = float(client.get_asset_balance(asset='USDT')['free'])
            amount = balance * 0.95
            logger.info(f"Using {amount} USDT from available balance")
        
        # Calculate quantity based on amount
        quantity = amount / current_price
        
        # Get symbol info for quantity precision using cached function
        symbol_info = get_symbol_info(SYMBOL)
        quantity_precision = 0
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                quantity_precision = len(str(float(filter['stepSize'])).rstrip('0').split('.')[-1])
                break
        
        # Round quantity to proper precision
        quantity = round(quantity, quantity_precision)
        logger.info(f"Calculated quantity: {quantity}")
        
        # Place market buy order
        order = client.create_order(
            symbol=SYMBOL,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        
        logger.info(f"Order placed successfully: {order}")
        return {
            "status": "success",
            "order": order,
            "message": f"Successfully bought {quantity} {SYMBOL} at market price"
        }
            
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"status": "error", "message": str(e)}

async def execute_sell():
    try:
        # Get asset balance
        asset = SYMBOL.replace('USDT', '')
        balance = float(client.get_asset_balance(asset=asset)['free'])
        logger.info(f"Available balance: {balance} {asset}")
        
        # Get symbol info for quantity precision using cached function
        symbol_info = get_symbol_info(SYMBOL)
        quantity_precision = 0
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                quantity_precision = len(str(float(filter['stepSize'])).rstrip('0').split('.')[-1])
                break
        
        # Round quantity to proper precision
        quantity = round(balance, quantity_precision)
        logger.info(f"Calculated quantity: {quantity}")
        
        # Place market sell order
        order = client.create_order(
            symbol=SYMBOL,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        
        logger.info(f"Order placed successfully: {order}")
        return {
            "status": "success",
            "order": order,
            "message": f"Successfully sold {quantity} {SYMBOL} at market price"
        }
            
    except BinanceAPIException as e:
        logger.error(f"Binance API error: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/buy")
async def buy_endpoint(request: TradeRequest):
    try:
        current_rsi, current_price = await get_market_data()
        
        if current_rsi is None or current_price is None:
            raise HTTPException(
                status_code=503,
                detail="Failed to get market data. Please try again."
            )
        
        # Check if RSI is in buy zone
        if current_rsi <= RSI_OVERSOLD:
            result = await execute_buy(request.amount)
            return {
                "status": "success",
                "rsi": current_rsi,
                "price": current_price,
                "trade_result": result,
                "signal": "BUY",
                "message": f"RSI ({current_rsi:.2f}) indicates buy signal"
            }
        else:
            return {
                "status": "skipped",
                "message": f"RSI ({current_rsi:.2f}) is not in buy zone (should be <= {RSI_OVERSOLD})",
                "rsi": current_rsi,
                "price": current_price,
                "signal": "HOLD"
            }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in buy endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.post("/toggle-auto-trade")
async def toggle_auto_trade():
    global AUTO_TRADE_ENABLED
    AUTO_TRADE_ENABLED = not AUTO_TRADE_ENABLED
    return {
        "status": "success",
        "auto_trade_enabled": AUTO_TRADE_ENABLED
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            # Get current RSI and price
            current_rsi, current_price = await get_market_data()
            
            if current_rsi is not None:
                # Determine trading signal
                signal = None
                if current_rsi <= RSI_OVERSOLD:
                    signal = "BUY"
                elif current_rsi >= RSI_OVERBOUGHT:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                # Execute trade if signal exists and auto-trade is enabled
                trade_result = None
                if signal in ["BUY", "SELL"] and AUTO_TRADE_ENABLED:
                    if signal == "BUY":
                        trade_result = await execute_buy()
                    elif signal == "SELL":
                        trade_result = await execute_sell()
                
                # Send data to frontend
                await websocket.send_json({
                    "rsi": current_rsi,
                    "price": current_price,
                    "signal": signal,
                    "trade_result": trade_result,
                    "auto_trade_enabled": AUTO_TRADE_ENABLED,
                    "message": f"RSI: {current_rsi:.2f} - Signal: {signal}"
                })
            
            # Wait for 1 minute before next update
            await asyncio.sleep(60)
            
        except Exception as e:
            print(f"WebSocket error: {e}")
            await websocket.send_json({
                "error": str(e)
            })
            await asyncio.sleep(5)

@app.get("/status")
async def get_status():
    try:
        current_rsi, current_price = await get_market_data()
        
        # Determine signal based on RSI
        signal = "HOLD"
        if current_rsi <= RSI_OVERSOLD:
            signal = "BUY"
        elif current_rsi >= RSI_OVERBOUGHT:
            signal = "SELL"
            
        return {
            "status": "operational",
            "rsi": current_rsi,
            "price": current_price,
            "symbol": SYMBOL,
            "auto_trade_enabled": AUTO_TRADE_ENABLED,
            "signal": signal,
            "message": f"RSI: {current_rsi:.2f} - Signal: {signal}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
