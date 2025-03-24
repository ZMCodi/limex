import pandas as pd

class TAEngine:

    def calculate_ma(self, data: pd.Series, ewm: bool, param_type: str, 
                    param: float, name: str) -> pd.Series:
        if ewm:
            return data.ewm(**{f'{param_type}': param}).mean()
        return data.rolling(window=param).mean()

    def calculate_rsi(self, data: pd.Series, window: int, name: str) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, min_periods=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data: pd.Series, windows: list[int], name: str) -> pd.DataFrame:
        results = pd.DataFrame(index=data.index)

        alpha_fast = 2 / (windows[0] + 1)
        alpha_slow = 2 / (windows[1] + 1)
        alpha_signal = 2 / (windows[2] + 1)

        results['macd'] = (self.calculate_ma(data, True, 'alpha', alpha_fast, name)
                            - self.calculate_ma(data, True, 'alpha', alpha_slow, name))

        results['signal_line'] = self.calculate_ma(results['macd'], True, 'alpha', alpha_signal, name)

        results['macd_hist'] = results['signal_line'] - results['macd']

        return results

    def rolling_std(self, data: pd.Series, ewm: bool, param_type: str, 
                    param: float, name: str) -> pd.Series:
        if ewm:
            return data.ewm(**{f'{param_type}': param}).std()
        return data.rolling(window=param).std()

    def calculate_bb(self, data: pd.Series, window: int, num_std: float, 
                    name: str) -> pd.DataFrame:
        results = pd.DataFrame(index=data.index)

        results['sma'] = self.calculate_ma(data, False, 'window', window, name)
        std = self.rolling_std(data, False, 'window', window, name)
        results['bol_up'] = results['sma'] + num_std * std
        results['bol_down'] = results['sma'] - num_std * std

        return results
