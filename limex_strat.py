import pandas as pd
import numpy as np

from ta import TAEngine
import signal_gen as sg
from utils import get_access_token, fetch_price_data
from abc import ABC, abstractmethod
import multiprocessing as mp
import scipy.optimize as sco
from itertools import product

token = get_access_token()

class Strategy(ABC):
    def __init__(self, symbol, data=None, days_back=3, period='minute_5'):
        self.symbol = symbol
        self.access_token = token
        if data is None:
            self.data = fetch_price_data(self.symbol, self.access_token, days_back, period)
        else:
            self.data = data
        self.ta = TAEngine()

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def change_params(self):
        pass

    def backtest(self, start=None, end=None):
        if start is None:
            start = self.data.index[0]
        if end is None:
            end = self.data.index[-1]
        df = self.data.loc[start:end]
        return np.exp(df[['returns', 'strategy']].sum()) - 1

    @classmethod
    def _optimize_helper(cls, params, start=None, end=None):
        strat = cls(**params)
        return strat.backtest(start, end)

    @property
    def num_signals(self):
        return np.sum(np.where(self.data['signal'].shift(1) != self.data['signal'], 1, 0)) 


class MA_C(Strategy):
    def __init__(self, symbol, data=None, days_back=3, period='minute_5', short=20, long=50):
        super().__init__(symbol, data, days_back, period)
        self.short = short
        self.long = long
        self.get_data()

    def get_data(self):
        df = self.data
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        df['short'] = self.ta.calculate_ma(df['close'], False, 'window', self.short, 'short')
        df['long'] = self.ta.calculate_ma(df['close'], False, 'window', self.long, 'long')

        df['signal'] = sg.ma_crossover(df['short'], df['long'])
        df['strategy'] = df['signal'].shift(1) * df['returns']

        self.data = df

    def change_params(self, short=None, long=None):
        if short is not None:
            self.short = int(short)
        if long is not None:
            self.long = int(long)
        self.get_data()

    def optimize(self, start=None, end=None):
        short_range = range(20, 61, 5)
        long_range = range(100, 281, 5)

        results = []
        params = {'symbol': self.symbol, 'data': self.data}
        for short, long in product(short_range, long_range):
            params['short'] = short
            params['long'] = long
            res = self._optimize_helper(params, start, end)
            results.append((short, long, res['returns'], res['strategy']))

        results = pd.DataFrame(results, columns=['short', 'long', 'returns', 'strategy'])
        results['net'] = results['strategy'] - results['returns']
        results = results.sort_values('net', ascending=False)
        return results


class RSI(Strategy):
    def __init__(self, symbol, data=None, days_back=3, period='minute_5', window=14,
                 overbought=70, oversold=30, mrev=50, weights=[1/2, 1/2], vote_thresh=0.):
        super().__init__(symbol, data, days_back, period)
        self.window = window
        self.overbought = overbought
        self.oversold = oversold
        self.mrev = mrev
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()
        self.vote_thresh = vote_thresh
        self.get_data()

    def get_data(self):
        df = self.data
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        df['rsi'] = self.ta.calculate_rsi(df['close'], self.window, 'rsi')

        df['signal'] = sg.rsi(df['rsi'], df['close'], self.overbought, self.oversold,
                              're', ['crossover', 'divergence'], 'weighted', self.vote_thresh,
                              self.weights, self.mrev)
        df['strategy'] = df['signal'].shift(1) * df['returns']

        self.data = df

    def change_params(self, window=None, overbought=None, oversold=None, mrev=None, weights=None, vote_thresh=None):
        if window is not None:
            self.window = int(window)
        if overbought is not None:
            self.overbought = overbought
        if oversold is not None:
            self.oversold = oversold
        if mrev is not None:
            self.mrev = mrev
        if weights is not None:
            self.weights = np.array(weights)
            self.weights /= self.weights.sum()
        if vote_thresh is not None:
            self.vote_thresh = vote_thresh
        self.get_data()

    def optimize(self, start=None, end=None):
        ovb_range = range(60, 81, 5)
        osd_range = range(20, 41, 5)
        mrev_range = range(40, 61, 5)
        window_range = range(10, 31, 5)

        results = []
        params = {'symbol': self.symbol, 'data': self.data}
        for window, ovb, osd, mrev in product(window_range, ovb_range, osd_range, mrev_range):
            params['window'] = window
            params['overbought'] = ovb
            params['oversold'] = osd
            params['mrev'] = mrev

            res = self._optimize_helper(params, start, end)
            results.append((window, ovb, osd, mrev, res['returns'], res['strategy']))

        results = pd.DataFrame(results, columns=['window', 'overbought', 'oversold', 'mrev', 'returns', 'strategy'])
        results['net'] = results['strategy'] - results['returns']
        results = results.sort_values('net', ascending=False)
        return results



class MACD(Strategy):
    def __init__(self, symbol, data=None, days_back=3, period='minute_5', fast=12, slow=26, signal=9,
                 weights=[1/5]*5, vote_thresh=0.):
        super().__init__(symbol, data, days_back, period)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()
        self.vote_thresh = vote_thresh
        self.get_data()
    
    def get_data(self):
        df = self.data
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        df[['macd', 'signal_line', 'macd_hist']] = self.ta.calculate_macd(
            df['close'], [self.fast, self.slow, self.signal], 'macd'
        )

        df['signal'] = sg.macd(
            df['macd_hist'], df['macd'], df['close'],
            ['crossover', 'divergence', 'hidden divergence', 'momentum', 'double peak/trough'],
            'weighted', self.vote_thresh, self.weights
        )
        df['strategy'] = df['signal'].shift(1) * df['returns']

        self.data = df

    def change_params(self, fast=None, slow=None, signal=None, weights=None, vote_thresh=None):
        if fast is not None:
            self.fast = int(fast)
        if slow is not None:
            self.slow = int(slow)
        if signal is not None:
            self.signal = int(signal)
        if weights is not None:
            self.weights = np.array(weights)
            self.weights /= self.weights.sum()
        if vote_thresh is not None:
            self.vote_thresh = vote_thresh
        self.get_data()

    def optimize(self, start=None, end=None):
        fast_range = range(8, 21, 1)
        slow_range = range(21, 35, 1)
        signal_range = range(5, 15, 1)

        results = []
        params = {'symbol': self.symbol, 'data': self.data}
        for fast, slow, signal in product(fast_range, slow_range, signal_range):
            params['fast'] = fast
            params['slow'] = slow
            params['signal'] = signal

            res = self._optimize_helper(params, start, end)
            results.append((fast, slow, signal, res['returns'], res['strategy']))

        results = pd.DataFrame(results, columns=['fast', 'slow', 'signal', 'returns', 'strategy'])
        results['net'] = results['strategy'] - results['returns']
        results = results.sort_values('net', ascending=False)
        return results
    


class BB(Strategy):
    def __init__(self, symbol, data=None, days_back=3, period='minute_5', window=20, width=2,
                 weights=[1/6]*6, vote_thresh=0.):
        super().__init__(symbol, data, days_back, period)
        self.window = window
        self.width = width
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()
        self.vote_thresh = vote_thresh
        self.get_data()

    def get_data(self):
        df = self.data
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        df[['sma', 'bol_up', 'bol_down']] = self.ta.calculate_bb(
            df['close'], self.window, self.width, 'bb'
        )
        df['signal'] = sg.bb(
            df['close'], df['bol_up'], df['bol_down'],
            ['bounce', 'double', 'walks', 'squeeze', 'breakout', '%B'],
            'weighted', self.vote_thresh, self.weights
        )
        df['strategy'] = df['signal'].shift(1) * df['returns']

        self.data = df

    def change_params(self, window=None, width=None, weights=None, vote_thresh=None):
        if window is not None:
            self.window = int(window)
        if width is not None:
            self.width = width
        if weights is not None:
            self.weights = np.array(weights)
            self.weights /= self.weights.sum()
        if vote_thresh is not None:
            self.vote_thresh = vote_thresh
        self.get_data()

    def optimize(self, start=None, end=None):
        window_range = range(10, 51, 5)
        width_range = np.arange(1, 3.1, 0.1)

        results = []
        params = {'symbol': self.symbol, 'data': self.data}
        for window, width in product(window_range, width_range):
            params['window'] = window
            params['width'] = width

            res = self._optimize_helper(params, start, end)
            results.append((window, width, res['returns'], res['strategy']))

        results = pd.DataFrame(results, columns=['window', 'width', 'returns', 'strategy'])
        results['net'] = results['strategy'] - results['returns']
        results = results.sort_values('net', ascending=False)
        return results


class Combined(Strategy):
    def __init__(self, symbol, data=None, days_back=3, period='minute_5', strategies=None,
                 weights=[1/4]*4, vote_thresh=0.):
        super().__init__(symbol, data, days_back, period)
        if strategies is None:
            self.strategies = [strat(symbol, days_back=days_back, period=period) for strat in [MA_C, RSI, MACD, BB]]
        else:
            self.strategies = strategies
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()
        self.vote_thresh = vote_thresh
        self.get_data()

    def get_data(self, signals=False, optimize=True):
        df = self.data
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        if not signals:
            signals = pd.DataFrame(index=df.index)
            for strat in self.strategies:
                if optimize:
                    # optimize and change parameters
                    opt = strat.optimize().iloc[0]
                    opt = opt.drop(['returns', 'strategy', 'net']).to_dict()
                    strat.change_params(**opt)

                signals[strat.__class__.__name__] = strat.data['signal']
                df[f'{strat.__class__.__name__}_signal'] = strat.data['signal']
        else:
            signals = df.filter(regex='_signal$')

        df['signal'] = sg.vote(signals, self.vote_thresh, self.weights)
        df['strategy'] = df['signal'].shift(1) * df['returns']
        self.data = df

    def change_params(self, strategies=None, weights=None, vote_thresh=None):
        if strategies is not None:
            self.strategies = strategies
        if weights is not None:
            self.weights = np.array(weights)
            self.weights /= self.weights.sum()
        if vote_thresh is not None:
            self.vote_thresh = vote_thresh
        self.get_data(signals=True)

    def optimize(self, start=None, end=None, threshold_range=None):
        old_params = {'weights': self.weights, 'vote_thresh': self.vote_thresh}

        n_weights = len(self.weights)
        if threshold_range is None:
            threshold_range = np.arange(0.2, 0.9, 0.1)

        weight_combinations = product(np.arange(0.1, 1.1, 0.1), repeat=n_weights)
        weight_combinations = [w for w in weight_combinations if np.isclose(np.sum(w), 1.0)]
        threshold_combinations = threshold_range

        best_value = float('inf')
        best_params = None

        for weights in weight_combinations:
            for threshold in threshold_combinations:
                self.change_params(weights=np.array(weights), vote_thresh=threshold)
                res = self.backtest(start=start, end=end)
                value = -res['strategy'].sum()
                if value < best_value:
                    best_value = value
                    best_params = (weights, threshold)

        opt_weights, opt_threshold = best_params

        self.change_params(weights=opt_weights, vote_thresh=opt_threshold)
        res = self.backtest(start=start, end=end)
        res.rename({'strategy': 'strategy_returns', 'returns': 'hold_returns'}, inplace=True)
        res['net'] = res['strategy_returns'] - res['hold_returns']

        self.change_params(**old_params)
        return {
            'weights': [float(w) for w in opt_weights],
            'vote_thresh': float(opt_threshold),
            'results': res.to_dict(),
        }
