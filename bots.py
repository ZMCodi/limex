import time
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from limex_strat import Combined, MA_C, RSI, MACD, BB
from utils import get_access_token, fetch_price_data, execute_trade
import pytz


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
        
        # Set up access token management
        self.access_token = None
        self.token_timestamp = 0  # When the token was last obtained
        self.refresh_access_token()  # Get initial token
        
        # Trading state - these will be updated if a log file exists
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
        
        # Check if log file exists and restore state if it does
        self.restore_state_from_log()
        
        # Check if params file exists and load or optimize strategy accordingly
        self.initialize_strategy()
    
    def refresh_access_token(self):
        """Get a new access token or refresh existing one if expired"""
        current_time = time.time()
        # Tokens typically expire after 24 hours (86400 seconds)
        # Refresh if no token or if token is older than 23 hours (82800 seconds)
        if self.access_token is None or (current_time - self.token_timestamp) > 82800:
            print(f"Getting new access token for {self.symbol}...")
            self.access_token = get_access_token()
            self.token_timestamp = current_time
            print(f"New access token obtained for {self.symbol}")
    
    def restore_state_from_log(self):
        """Restore trading state from existing log file if available"""
        if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
            try:
                # Read the log file
                log_df = pd.read_csv(self.log_file)
                
                if not log_df.empty:
                    # Get the most recent entry
                    last_entry = log_df.iloc[-1]
                    
                    # Restore state from the last entry
                    self.current_position = int(last_entry['quantity']) if last_entry['action'] == 'BUY' else 0
                    self.cash_balance = float(last_entry['cash_balance'])
                    self.portfolio_value = float(last_entry['portfolio_value'])
                    self.last_signal = int(last_entry['signal'])
                    
                    print(f"Restored state for {self.symbol} from log file:")
                    print(f"  - Current position: {self.current_position} shares")
                    print(f"  - Cash balance: ${self.cash_balance:.2f}")
                    print(f"  - Portfolio value: ${self.portfolio_value:.2f}")
                    print(f"  - Last signal: {self.last_signal}")
                else:
                    # Create the header if file exists but is empty
                    with open(self.log_file, 'w') as f:
                        f.write("timestamp,symbol,action,signal,price,quantity,cash_balance,portfolio_value\n")
            except Exception as e:
                print(f"Error restoring state from log file for {self.symbol}: {e}")
                # Create a new log file with header
                with open(self.log_file, 'w') as f:
                    f.write("timestamp,symbol,action,signal,price,quantity,cash_balance,portfolio_value\n")
        else:
            # Create a new log file with header
            with open(self.log_file, 'w') as f:
                f.write("timestamp,symbol,action,signal,price,quantity,cash_balance,portfolio_value\n")
    
    def initialize_strategy(self):
        """Initialize the strategy by loading existing parameters or optimizing"""
        params_file = os.path.join('params', f"{self.symbol}_strategy_params.json")
        
        if os.path.exists(params_file):
            try:
                # Load existing parameters
                with open(params_file, 'r') as f:
                    params_data = json.load(f)
                
                if self.symbol in params_data:
                    symbol_params = params_data[self.symbol]
                    
                    # Create Combined strategy
                    self.strategy = Combined(self.symbol, days_back=self.days_back, period=self.period)
                    
                    # Apply parameters to Combined strategy
                    self.strategy.change_params(
                        weights=symbol_params.get('weights', [0.25, 0.25, 0.25, 0.25]),
                        vote_thresh=symbol_params.get('vote_thresh', 0.0)
                    )
                    
                    # Apply individual strategy parameters if they exist
                    if 'individual_params' in symbol_params:
                        ind_params = symbol_params['individual_params']
                        
                        # Apply to each strategy, assuming order is [MA_C, RSI, MACD, BB]
                        for i, strat in enumerate(self.strategy.strategies):
                            strat_name = strat.__class__.__name__
                            if strat_name in ind_params:
                                strat.change_params(**ind_params[strat_name])
                    
                    # Set last optimization timestamp
                    self.last_optimization = symbol_params.get('timestamp', time.time())
                    
                    print(f"Loaded existing parameters for {self.symbol}")
                    print(f"  - Weights: {self.strategy.weights}")
                    print(f"  - Threshold: {self.strategy.vote_thresh}")
                    print(f"  - Last optimization: {datetime.fromtimestamp(self.last_optimization)}")
                    for s in self.strategy.strategies:
                        print(s.params)
                    
                    return True
            except Exception as e:
                print(f"Error loading parameters for {self.symbol}: {e}")
        
        # If we get here, either no parameters exist or there was an error loading them
        print(f"No existing parameters found for {self.symbol}. Optimizing...")
        return self.optimize_strategy()
    
    def optimize_strategy(self):
        """Create and optimize a new strategy instance"""
        print(f"Optimizing strategy for {self.symbol}...")
        
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
            print('Optimization complete')
            print(f"Optimization complete for {self.symbol}")
            print(f"  - Weights: {self.strategy.weights}")
            print(f"  - Threshold: {self.strategy.vote_thresh}")
            
            return True
        
        except Exception as e:
            print(f"Error during optimization for {self.symbol}: {e}")
            return False
    
    def save_parameters(self, optimization_results, filename):
        """Save strategy parameters to file"""
        # Ensure params directory exists
        if not os.path.exists('params'):
            os.makedirs('params')

        # Add params/ prefix to filename
        filepath = os.path.join('params', filename)
        
        # Collect individual strategy parameters
        individual_params = {}
        for strat in self.strategy.strategies:
            strat_name = strat.__class__.__name__
            individual_params[strat_name] = strat.params
            if 'weights' in strat.params:
                individual_params[strat_name]['weights'] = list(individual_params[strat_name]['weights'])
        
        params = {
            "symbol": self.symbol,
            "timestamp": self.last_optimization,
            "weights": [float(w) for w in optimization_results['weights']],
            "vote_thresh": float(optimization_results['vote_thresh']),
            "individual_params": individual_params
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
            print(f"Cannot execute {side} for {self.symbol}: quantity must be positive")
            return False
        
        # Ensure access token is fresh
        self.refresh_access_token()
        
        try:
            execute_trade(self.access_token, self.symbol, side, quantity)
            print(f"Executed {side} for {self.symbol}: {quantity} shares at ${price:.2f}")
            return True
        except Exception as e:
            print(f"Error executing {side} for {self.symbol}: {e}")
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
        
        print(f"{action} {self.symbol}: {quantity} shares at ${price:.2f}")
        print(f"  - New position: {self.current_position} shares")
        print(f"  - Cash balance: ${self.cash_balance:.2f}")
        print(f"  - Portfolio value: ${self.portfolio_value:.2f}")
    
    def process_signal(self, signal, current_price):
        """Process a trading signal and execute appropriate trades"""
        # Only act if signal has changed
        if signal == self.last_signal:
            return
        
        # Calculate maximum shares we can buy with our cash
        max_shares_to_buy = int(self.cash_balance / current_price)
        
        if signal == 1 and self.last_signal != 1:
            print(f'buying for {self.symbol}')
            # BUY signal
            if max_shares_to_buy > 0:
                print(f'buying {max_shares_to_buy} shares')
                if self.execute_trade("buy", current_price, max_shares_to_buy):
                    self.update_portfolio("BUY", signal, current_price, max_shares_to_buy)
                    self.last_signal = signal
        
        elif signal == -1 and self.last_signal != -1:
            # SELL signal - sell all current position
            print(f'selling for {self.symbol}')
            if self.current_position > 0:
                if self.execute_trade("sell", current_price, self.current_position):
                    self.update_portfolio("SELL", signal, current_price, self.current_position)
                    self.last_signal = signal
        
        # Update signal state
        self.last_signal = signal
    
    def run(self, refresh_interval=60):
        """Run the trading bot"""
        print(f"Starting trading bot for {self.symbol} with ${self.allocation:.2f} allocation")
        
        try:
            while True:
                current_time = time.time()
                
                # Refresh token if needed
                self.refresh_access_token()
                
                # Check if market is open
                if not self.is_market_open():
                    next_open = self.time_until_next_market_open()
                    print(f"Market closed for {self.symbol}. Sleeping until next market open in {next_open:.1f} hours")
                    # Sleep until next check time (check every 15 minutes when market is closed)
                    time.sleep(min(900, next_open * 3600))  # Sleep for 15 minutes or until market opens
                    continue
                
                # Check if we need to re-optimize (every reoptimize_days)
                seconds_since_optimization = current_time - self.last_optimization
                if seconds_since_optimization > (self.reoptimize_days * 24 * 3600):
                    print(f"Time to re-optimize strategy for {self.symbol}")
                    self.optimize_strategy()
                
                # Fetch latest data
                try:
                    print(f"Fetching data for {self.symbol}")
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
                        print(f"No data received for {self.symbol}")
                
                except Exception as e:
                    print(f"Error processing data for {self.symbol}: {e}")
                
                # Sleep for refresh interval
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print(f"Trading bot for {self.symbol} stopped by user")
        except Exception as e:
            print(f"Critical error in trading bot for {self.symbol}: {e}")
            raise

    def is_market_open(self):
        """Check if the NYSE market is currently open"""
        # Define NYSE trading hours (9:30 AM - 4:00 PM Eastern Time)
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Check if it's a weekend
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if it's during trading hours (9:30 AM - 4:00 PM)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close

    def time_until_next_market_open(self):
        """Calculate time in hours until the next market open"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Figure out the next market open time
        if now.weekday() >= 5:  # Weekend
            # If Saturday (5), we need to wait until Monday
            days_to_add = 7 - now.weekday()
        else:  # Weekday
            if now.hour < 9 or (now.hour == 9 and now.minute < 30):
                # Before market open on a weekday
                days_to_add = 0
            else:
                # After market hours on a weekday
                if now.weekday() == 4:  # Friday
                    days_to_add = 3  # Wait until Monday
                else:
                    days_to_add = 1  # Wait until tomorrow
        
        # Calculate the next open time
        next_day = now + timedelta(days=days_to_add)
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If same day and currently before market open
        if days_to_add == 0:
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Calculate the difference in hours
        diff = next_open - now
        hours_until_open = diff.total_seconds() / 3600
        
        return hours_until_open
