#!/usr/bin/env python3
"""
Main script to launch trading bots for multiple symbols.
This is the entry point for the trading system.
"""

from bots import run_trading_bots

if __name__ == "__main__":
    # Configure the symbols you want to trade
    symbols_to_trade = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Google
        "AMZN",  # Amazon
        "META",  # Meta (Facebook)
        "TSLA",  # Tesla
        "NVDA",  # NVIDIA
        "AMD",   # AMD
        "JPM",   # JPMorgan Chase
        "V"      # Visa
    ]
    
    # Total capital to allocate across all symbols
    TOTAL_CAPITAL = 1000
    
    print(f"Starting trading system with ${TOTAL_CAPITAL} capital")
    print(f"Trading {len(symbols_to_trade)} symbols: {', '.join(symbols_to_trade)}")
    print(f"Capital per symbol: ${TOTAL_CAPITAL / len(symbols_to_trade)}")
    print("Press Ctrl+C to stop all trading bots")
    
    # Launch the trading bots
    run_trading_bots(symbols_to_trade, total_capital=TOTAL_CAPITAL)
