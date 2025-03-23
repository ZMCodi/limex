import pandas as pd

class TAEngine:

    def __init__(self):
        self.cache = {}

    def calculate_ma(self, data: pd.Series, ewm: bool, param_type: str, 
                    param: float, name: str) -> pd.Series:
        
        key = f'ma_{param_type}={param}, {name=}'
        if key not in self.cache:
            if ewm:
                self.cache[key] = data.ewm(**{f'{param_type}': param}).mean()
            else:
                self.cache[key] = data.rolling(window=param).mean()

        return self.cache[key]

    def calculate_rsi(self, data: pd.Series, window: int, name: str) -> pd.Series:
        
        key = f'rsi_window={window}, {name=}'
        if key not in self.cache:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
            rs = gain / loss
            self.cache[key] = 100 - (100 / (1 + rs))

        return self.cache[key]

    def calculate_macd(self, data: pd.Series, windows: list[int], name: str) -> pd.DataFrame:
        
        key = f'macd_{windows=}, {name=}'
        if key not in self.cache:
            results = pd.DataFrame(index=data.index)

            alpha_fast = 2 / (windows[0] + 1)
            alpha_slow = 2 / (windows[1] + 1)
            alpha_signal = 2 / (windows[2] + 1)

            results['macd'] = (self.calculate_ma(data, True, 'alpha', alpha_fast, name)
                                - self.calculate_ma(data, True, 'alpha', alpha_slow, name))

            results['signal_line'] = self.calculate_ma(results['macd'], True, 'alpha', alpha_signal, name)

            results['macd_hist'] = results['signal_line'] - results['macd']

            self.cache[key] = results

        return self.cache[key]

    def rolling_std(self, data: pd.Series, ewm: bool, param_type: str, 
                    param: float, name: str) -> pd.Series:
       
        key = f'rol_std_{param_type}={param}, {name=}'
        if key not in self.cache:
            if ewm:
                self.cache[key] = data.ewm(**{f'{param_type}': param}).std()
            else:
                self.cache[key] = data.rolling(window=param).std()

        return self.cache[key]

    def calculate_bb(self, data: pd.Series, window: int, num_std: float, 
                    name: str) -> pd.DataFrame:
        
        key = f'bb_{window=}_{num_std=}, {name}'
        if key not in self.cache:
            results = pd.DataFrame(index=data.index)

            results['sma'] = self.calculate_ma(data, False, 'window', window, name)
            std = self.rolling_std(data, False, 'window', window, name)
            results['bol_up'] = results['sma'] + num_std * std
            results['bol_down'] = results['sma'] - num_std * std

            self.cache[key] = results

        return self.cache[key]
