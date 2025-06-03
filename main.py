from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from ta.momentum import RSIIndicator
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime
import requests
import time
from pydantic import BaseModel
from typing import Optional

# Load .env
load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

app = FastAPI(
    title="Crypto Trading Bot",
    description="A FastAPI application for automated crypto trading using RSI strategy",
    version="1.0.0"
)

# Pydantic models for request validation
class TradeRequest(BaseModel):
    symbol: str = "BTCUSDT"
    quantity: float = 0.001

class RSIRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1m"
    limit: int = 100

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None

def get_time_offset():
    """Get the time offset between local and Binance server time"""
    try:
        server_time = requests.get('https://api.binance.com/api/v3/time').json()['serverTime']
        local_time = int(time.time() * 1000)
        return server_time - local_time
    except Exception as e:
        print(f"Error getting time offset: {str(e)}")
        return 0

def create_client_with_time_patch():
    """Create a Binance client with monkey-patched timestamp function"""
    try:
        # Get initial time offset
        offset = get_time_offset()
        print(f"Initial time offset: {offset}ms")

        # Create custom timestamp function with offset
        def custom_timestamp():
            return int(time.time() * 1000 + offset)

        # Initialize client
        client = Client(api_key, api_secret, testnet=True)
        
        # Monkey-patch the timestamp function
        client._get_timestamp = custom_timestamp
        
        # Test connection
        client.ping()
        print(f"✅ Connected to Binance Testnet with custom timestamp")
        return client
    except Exception as e:
        print(f"❌ Binance connection error: {str(e)}")
        return None

def initialize_binance_client():
    """Initialize the Binance client with time synchronization"""
    global client
    client = create_client_with_time_patch()
    return client

client = initialize_binance_client()


async def calculate_rsi(symbol="BTCUSDT", interval="1m", limit=100):
    global client
    try:
        if not client:
            client = initialize_binance_client()
            if not client:
                raise HTTPException(status_code=503, detail="Binance API unavailable")

        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'])
        df['close'] = df['close'].astype(float)
        rsi = RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] RSI: {rsi:.2f}")
        return rsi
    except Exception as e:
        print(f"RSI Calculation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_auto_trade(symbol: str, side: str, quantity: float):
    global client
    if not client:
        client = initialize_binance_client()
        if not client:
            return {"status": "error", "message": "Binance API not available"}

    for attempt in range(3):
        try:
            # Create order with increased recvWindow
            order = client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity,
                recvWindow=15000  # Increased to 15 seconds
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Trade executed: {order}")
            return {"status": "success", "order": order}

        except BinanceAPIException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Trade attempt {attempt + 1} failed: {str(e)}")
            if "Timestamp" in str(e):
                # If timestamp error, reinitialize client with fresh time sync
                print("Reinitializing client due to timestamp error...")
                client = initialize_binance_client()
                await asyncio.sleep(1)
                continue
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "Max retries exceeded"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            rsi = await calculate_rsi()
            signal, trade_result = None, None
            
            # More aggressive trading conditions
            if rsi <= 48:  # Buy when RSI is below 48
                signal = "BUY"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔵 Buy signal (RSI: {rsi:.2f})")
                trade_result = await execute_auto_trade("BTCUSDT", Client.SIDE_BUY, 0.001)
            elif rsi >= 52:  # Sell when RSI is above 52
                signal = "SELL"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 Sell signal (RSI: {rsi:.2f})")
                trade_result = await execute_auto_trade("BTCUSDT", Client.SIDE_SELL, 0.001)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚪ No trade (RSI: {rsi:.2f})")
            
            await websocket.send_json({
                "rsi": rsi,
                "signal": signal,
                "trade_result": trade_result,
                "timestamp": datetime.now().strftime('%H:%M:%S')
            })
            
            # Reduced delay for more frequent trading
            await asyncio.sleep(3)  # Check every 3 seconds
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/")
async def root():
    """Root endpoint that returns basic API information"""
    return {
        "name": "Crypto Trading Bot",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/rsi")
async def get_rsi(request: RSIRequest):
    """Get current RSI value for a symbol"""
    try:
        rsi = await calculate_rsi(
            symbol=request.symbol,
            interval=request.interval,
            limit=request.limit
        )
        return {
            "symbol": request.symbol,
            "rsi": rsi,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade/buy")
async def buy_crypto(request: TradeRequest):
    """Execute a buy order"""
    result = await execute_auto_trade(
        symbol=request.symbol,
        side=Client.SIDE_BUY,
        quantity=request.quantity
    )
    return result

@app.post("/trade/sell")
async def sell_crypto(request: TradeRequest):
    """Execute a sell order"""
    result = await execute_auto_trade(
        symbol=request.symbol,
        side=Client.SIDE_SELL,
        quantity=request.quantity
    )
    return result

@app.get("/status")
async def get_status():
    """Get the current status of the trading bot"""
    try:
        if not client:
            return {"status": "disconnected", "message": "Binance client not initialized"}
        
        # Test connection
        client.ping()
        return {
            "status": "connected",
            "message": "Trading bot is running",
            "time_offset": client.TIME_OFFSET,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)