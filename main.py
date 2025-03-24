#!/usr/bin/env python3
"""
Main script to launch trading bots for a single symbol.
This is the entry point for the trading system.
"""

import sys
from bots import TradingBot

if __name__ == "__main__":
    # Check if symbol and allocation were provided as command line arguments
    if len(sys.argv) > 2:
        symbol = sys.argv[1]
        allocation = float(sys.argv[2])
    elif len(sys.argv) > 1:
        symbol = sys.argv[1]
        allocation = 300  # Default allocation if only symbol is provided
    else:
        # Default values if none provided
        symbol = "AAPL"
        allocation = 300
    
    print(f"Starting trading bot for {symbol} with ${allocation} capital")
    print("Press Ctrl+C to stop the trading bot")
    
    # Create and run a single bot
    bot = TradingBot(
        symbol=symbol,
        allocation=allocation,
        days_back=5,
        period='minute',
        reoptimize_days=5
    )
    
    # Run the bot
    bot.run()
