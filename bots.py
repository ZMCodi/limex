import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from limex_strat import Combined
from utils import get_access_token, fetch_price_data, execute_trade


class TradingBot:
    def __init__(self, symbol, allocation, days_back=5, period='minute', reoptimize_days=5):
        """
        Initialize the trading bot for a specific symbol
        
        Args:
            symbol (str): Trading symbol (e.g., 'AAPL')
            allocation (float): Amount of money allocated to this symbol
            days_back (int): Days of historical data to use for optimization
            period (str): Timeframe for data ('minute_1', 'minute_5', etc.)
            reoptimize_days (int): How often to re-optimize the strategy (in days)
        """
        self.symbol = symbol
        self.allocation = allocation
        self.days_back = days_back
        self.period = period
        self.reoptimize_days = reoptimize_days
        self.access_token = get_access_token()
        
        # Trading state
        self.current_position = 0  # Shares owned
        self.cash_balance = allocation  # Cash available
        self.portfolio_value = allocation  # Total portfolio value
        self.last_signal = 0  # No position initially
        
        # Strategy initialization
        self.strategy = None
        self.last_optimization = 0
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Initialize log file
        self.log_file = f"logs/{symbol}_trading_log.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("timestamp,symbol,action,signal,price,quantity,cash_balance,portfolio_value\n")
        
        # Initialize logger
        self.logger = self._initialize_logger()

    def _initialize_logger(self):
        """Initialize a logger for the trading bot"""
        logger = logging.getLogger(self.symbol)
        logger.setLevel(logging.INFO)
        
        # Create a file handler with a unique ID based on the date
        log_filename = f"logs/{self.symbol}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        
        return logger
    
    def optimize_strategy(self):
        """Create and optimize a new strategy instance"""
        self.logger.info(f"Optimizing strategy for {self.symbol}...")
        
        try:
            # Create Combined strategy with all sub-strategies
            self.strategy = Combined(self.symbol, days_back=self.days_back, period=self.period)
            
            # Run optimization to find best parameters
            optimization_results = self.strategy.optimize()
            
            # Apply optimized parameters
            self.strategy.change_params(
                weights=optimization_results['weights'],
                vote_thresh=optimization_results['vote_thresh']
            )
            
            # Update optimization timestamp
            self.last_optimization = time.time()
            
            # Save parameters to file
            self.save_parameters(optimization_results, f"{self.symbol}_strategy_params.json")
            
            self.logger.info(f"Optimization complete for {self.symbol}")
            self.logger.info(f"  - Weights: {self.strategy.weights}")
            self.logger.info(f"  - Threshold: {self.strategy.vote_thresh}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error during optimization for {self.symbol}: {e}")
            return False
    
    def save_parameters(self, optimization_results, filename):
        """Save strategy parameters to file"""
        # Ensure params directory exists
        if not os.path.exists('params'):
            os.makedirs('params')

        # Add params/ prefix to filename
        filepath = os.path.join('params', filename)
        
        params = {
            "symbol": self.symbol,
            "timestamp": self.last_optimization,
            "weights": [float(w) for w in optimization_results['weights']],
            "vote_thresh": float(optimization_results['vote_thresh'])
        }
        
        # Load existing parameters if file exists
        all_params = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                all_params = json.load(f)
        
        # Update parameters for this symbol
        all_params[self.symbol] = params
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(all_params, f, indent=4)

    def execute_trade(self, side, price, quantity):
        """Execute a trade using the trading API"""
        if quantity <= 0:
            self.logger.warning(f"Cannot execute {side} for {self.symbol}: quantity must be positive")
            return False
        
        try:
            execute_trade(self.access_token, self.symbol, side, quantity)
            self.logger.info(f"Executed {side} for {self.symbol}: {quantity} shares at ${price:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Error executing {side} for {self.symbol}: {e}")
            return False
    
    def update_portfolio(self, action, signal, price, quantity):
        """Update portfolio state after a trade and log the transaction"""
        timestamp = datetime.now()
        
        # Update portfolio based on the action
        if action == "BUY":
            self.cash_balance -= price * quantity
            self.current_position += quantity
        elif action == "SELL":
            self.cash_balance += price * quantity
            self.current_position -= quantity
        
        # Calculate new portfolio value
        self.portfolio_value = self.cash_balance + (self.current_position * price)
        
        # Log the transaction
        log_entry = f"{timestamp},{self.symbol},{action},{signal},{price},{quantity},{self.cash_balance},{self.portfolio_value}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        self.logger.info(f"{action} {self.symbol}: {quantity} shares at ${price:.2f}")
        self.logger.info(f"  - New position: {self.current_position} shares")
        self.logger.info(f"  - Cash balance: ${self.cash_balance:.2f}")
        self.logger.info(f"  - Portfolio value: ${self.portfolio_value:.2f}")
    
    def process_signal(self, signal, current_price):
        """Process a trading signal and execute appropriate trades"""
        # Only act if signal has changed
        if signal == self.last_signal:
            return
        
        # Calculate maximum shares we can buy with our cash
        max_shares_to_buy = int(self.cash_balance / current_price)
        
        if signal == 1 and self.last_signal != 1:
            # BUY signal
            if max_shares_to_buy > 0:
                if self.execute_trade("buy", current_price, max_shares_to_buy):
                    self.update_portfolio("BUY", signal, current_price, max_shares_to_buy)
                    self.last_signal = signal
        
        elif signal == -1 and self.last_signal != -1:
            # SELL signal - sell all current position
            if self.current_position > 0:
                if self.execute_trade("sell", current_price, self.current_position):
                    self.update_portfolio("SELL", signal, current_price, self.current_position)
                    self.last_signal = signal
        
        # Update signal state
        self.last_signal = signal
    
    def run(self, refresh_interval=60):
        """Run the trading bot"""
        self.logger.info(f"Starting trading bot for {self.symbol} with ${self.allocation:.2f} allocation")
        
        # Initial optimization
        if not self.optimize_strategy():
            self.logger.error(f"Failed to initialize strategy for {self.symbol}. Exiting.")
            return
        
        try:
            while True:
                current_time = time.time()
                
                # Check if we need to re-optimize (every reoptimize_days)
                seconds_since_optimization = current_time - self.last_optimization
                if seconds_since_optimization > (self.reoptimize_days * 24 * 3600):
                    self.logger.info(f"Time to re-optimize strategy for {self.symbol}")
                    self.optimize_strategy()
                
                # Fetch latest data
                try:
                    latest_data = fetch_price_data(self.symbol, self.access_token, 3, self.period)
                    
                    # Update strategy data
                    if self.strategy is not None and latest_data is not None and not latest_data.empty:
                        self.strategy.data = pd.concat([self.strategy.data, latest_data]).drop_duplicates()
                        
                        # Regenerate signals with optimized parameters
                        self.strategy.get_data(signals=False, optimize=False)
                        
                        # Get latest signal and price
                        latest_signal = self.strategy.data['signal'].iloc[-1]
                        current_price = self.strategy.data['close'].iloc[-1]
                        
                        # Process the signal
                        self.process_signal(latest_signal, current_price)
                    else:
                        self.logger.warning(f"No data received for {self.symbol}")
                
                except Exception as e:
                    self.logger.error(f"Error processing data for {self.symbol}: {e}")
                
                # Sleep for refresh interval
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            self.logger.info(f"Trading bot for {self.symbol} stopped by user")
        except Exception as e:
            self.logger.critical(f"Critical error in trading bot for {self.symbol}: {e}")
            raise


# Main function to run multiple trading bots
def run_trading_bots(symbols, total_capital=100000):
    """
    Run trading bots for multiple symbols
    
    Args:
        symbols (list): List of symbols to trade
        total_capital (float): Total capital to allocate across all symbols
    """
    # Calculate allocation per symbol
    num_symbols = len(symbols)
    allocation_per_symbol = total_capital / num_symbols
    
    # Create and start trading bots
    bots = []
    for symbol in symbols:
        bot = TradingBot(
            symbol=symbol,
            allocation=allocation_per_symbol,
            days_back=5,
            period='minute',
            reoptimize_days=5
        )
        bots.append(bot)
    
    # Run each bot in a separate process
    import multiprocessing as mp
    
    processes = []
    for bot in bots:
        p = mp.Process(target=bot.run)
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
