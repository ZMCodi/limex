

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Optional
import numba as nb

def ma_crossover(short: pd.Series, long: pd.Series) -> np.ndarray:

    return np.where(short > long, 1, -1)


def rsi(RSI: pd.Series, price: pd.Series, ub: float, lb: float, 
        exit: str, signal_type: list[str], method: str, 
        threshold: float, weights: Optional[list[float]] = None, 
        m_rev_bound: float = 50) -> pd.Series:

    signals = pd.DataFrame(index=RSI.index)

    if 'crossover' in signal_type:
        signals['crossover'] = rsi_crossover(RSI, ub, lb, exit, m_rev_bound)

    if 'divergence' in signal_type:
        signals['divergence'] = rsi_divergence(RSI, price, False)

    if 'hidden divergence' in signal_type:
        signals['hidden_div'] = rsi_divergence(RSI, price, True)

    if method == 'unanimous':
        threshold = .99
        weights = [1 / len(signals.columns)] * len(signals.columns)
    elif method == 'majority':
        threshold = 0
        weights = [1 / len(signals.columns)] * len(signals.columns)

    return vote(signals, threshold, weights)


def rsi_divergence(RSI: pd.Series, price: pd.Series, hidden: bool = False) -> pd.Series:

    return divergence_signals(RSI, *find_momentum_divergence(price, RSI, hidden=hidden))


def rsi_crossover(RSI: pd.Series, ub: float, lb: float, 
                 exit: str, m_rev_bound: Optional[float] = None) -> pd.Series:

    if exit == 're':
        signal = np.where(
            (RSI.shift(1) > ub) & (RSI < ub), -1, 
            np.where((RSI.shift(1) < lb) & (RSI > lb), 1, np.nan)
        )
        short_entries = (RSI.shift(1) > ub) & (RSI < ub)
    else:
        signal = np.where(
            RSI > ub, -1,
            np.where(RSI < lb, 1, np.nan)
        )
        short_entries = (RSI.shift(1) <= ub) & (RSI > ub)

    signal = pd.Series(signal, index=RSI.index)
    signal = fill(signal)

    if m_rev_bound is not None:
        mean_rev_points = (RSI <= m_rev_bound) & (signal == -1)
        groups = short_entries.cumsum()
        mean_rev_triggered = mean_rev_points.groupby(groups).cummax()
        signal = np.where(mean_rev_triggered, 1, signal)

    return signal


def macd(macd_hist: pd.Series, macd: pd.Series, price: pd.Series, 
         signal_type: list[str], method: str, threshold: float, 
         weights: Optional[list[float]] = None) -> pd.Series:

    signals = pd.DataFrame(index=macd_hist.index)

    if 'crossover' in signal_type:
        signals['crossover'] = np.where(macd_hist > 0, 1, -1)

    if 'divergence' in signal_type:
        signals['divergence'] = macd_divergence(macd, price, False)

    if 'hidden divergence' in signal_type:
        signals['hidden_div'] = macd_divergence(macd, price, True)

    if 'momentum' in signal_type:
        signals['momentum'] = macd_momentum(macd_hist)

    if 'double peak/trough' in signal_type:
        signals['double'] = macd_double(macd_hist)

    if method == 'unanimous':
        threshold = .99
        weights = [1 / len(signals.columns)] * len(signals.columns)
    elif method == 'majority':
        threshold = 0
        weights = [1 / len(signals.columns)] * len(signals.columns)

    return vote(signals, threshold, weights)


def macd_divergence(macd: pd.Series, price: pd.Series, hidden: bool = False) -> pd.Series:

    return divergence_signals(macd, *find_momentum_divergence(price, macd, hidden=hidden))


def macd_momentum(macd_hist: pd.Series) -> pd.Series:
    signal = np.where(
        (macd_hist.shift(1) < macd_hist)
        & (macd_hist.shift(1) < 0), 1,
        np.where(
            (macd_hist.shift(1) > macd_hist)
            & (macd_hist.shift(1) > 0), -1, np.nan
        )
    )

    signal = pd.Series(signal, index=macd_hist.index)
    return fill(signal)


def macd_double(macd_hist: pd.Series) -> pd.Series:

    return double_pattern_signals(macd_hist, *find_double_patterns(macd_hist))


def bb(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series,
      signal_type: list[str], method: str, threshold: float,
      weights: Optional[list[float]] = None) -> pd.Series:

    signals = pd.DataFrame(index=price.index)

    if 'bounce' in signal_type:
        signals['bounce'] = bb_bounce(price, bb_up, bb_down)

    if 'double' in signal_type:
        signals['double'] = bb_double(price, bb_up, bb_down)

    if 'walks' in signal_type:
        signals['walks'] = bb_walks(price, bb_up, bb_down)

    if 'squeeze' in signal_type:
        signals['squeeze'] = bb_squeeze(price, bb_up, bb_down)

    if 'breakout' in signal_type:
        signals['breakout'] = bb_breakout(price, bb_up, bb_down)

    if '%B' in signal_type:
        signals['%B'] = bb_pctB(price, bb_up, bb_down)

    if method == 'unanimous':
        threshold = .99
        weights = [1 / len(signals.columns)] * len(signals.columns)
    elif method == 'majority':
        threshold = 0
        weights = [1 / len(signals.columns)] * len(signals.columns)

    return vote(signals, threshold, weights)


def bb_bounce(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series) -> pd.Series:

    signal = pd.Series(np.where(
        (price.shift(1) > bb_up) & (price < bb_up), -1,
        np.where(
            (price.shift(1) < bb_down) & (price > bb_down), 1, np.nan
        )
    ), index=price.index)

    return fill(signal)


def bb_double(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series) -> pd.Series:

    rel_width = bb_up - bb_down
    hist = pd.Series(np.where(
        price > bb_up, (price - bb_up) / rel_width,
        np.where(price < bb_down, (price - bb_down) / rel_width, 0)
    ), index=price.index)

    return double_pattern_signals(price, *find_double_patterns(hist, 5, 15))


def bb_walks(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series,
            prox: float = 0.2, periods: int = 5) -> pd.Series:

    width = bb_up - bb_down
    close_upper = np.abs(price - bb_up) < width * prox
    close_lower = np.abs(price - bb_down) < width * prox

    upper_walk = close_upper.rolling(periods).sum() >= periods - 1
    lower_walk = close_lower.rolling(periods).sum() >= periods - 1

    walk = pd.Series(np.where(upper_walk, 1,
                    np.where(lower_walk, -1, np.nan)), index=price.index)

    return fill(walk)


def bb_squeeze(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series,
              aggressive: bool = False) -> pd.Series:

    width = bb_up - bb_down
    squeeze = width < width.rolling(20).quantile(0.2)

    if aggressive:
        ext = (width > width.shift(1)) & squeeze.shift(1)
    else:
        ext = ~squeeze & squeeze.shift(1)

    signal = pd.Series(np.where(
        ext & (price > price.shift(1)), 1,
        np.where(ext & (price < price.shift(1)), -1, np.nan)
    ), index=price.index)

    return fill(signal)


def bb_breakout(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series,
                threshold: float = 0.3) -> pd.Series:

    momentum = price.pct_change()
    mom_range = momentum.max() - momentum.min()

    signal = pd.Series(np.where(
        (price > bb_up) & (momentum > threshold * mom_range), 1,
        np.where((price < bb_down) & (momentum < -threshold * mom_range), -1, np.nan)
    ), index=price.index)

    return fill(signal)


def bb_pctB(price: pd.Series, bb_up: pd.Series, bb_down: pd.Series,
            overbought: float = 0.8, oversold: float = 0.2) -> pd.Series:
    pctB = (price - bb_down) / (bb_up - bb_down)
    signal = pd.Series(np.where(pctB > overbought, -1,
                      np.where(pctB < oversold, 1, np.nan)), index=price.index)

    return fill(signal)


def find_momentum_divergence(price: pd.Series, indicator: pd.Series,
                           distance_min: int = 7, distance_max: int = 25,
                           prominence: float = 0.05, hidden: bool = False,
                           is_rsi: bool = False, ub: float = 0.7,
                           lb: float = 0.3) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:

    price_peaks, _ = find_peaks(price, distance=distance_min,
                               prominence=prominence*(price.max() - price.min()))
    price_troughs, _ = find_peaks(-price, distance=distance_min,
                                 prominence=prominence*(price.max() - price.min()))
    
    ind_peaks, _ = find_peaks(indicator, distance=distance_min,
                             prominence=prominence*(indicator.max() - indicator.min()))
    ind_troughs, _ = find_peaks(-indicator, distance=distance_min,
                               prominence=prominence*(indicator.max() - indicator.min()))

    bearish_divs = []
    for i in range(len(price_peaks)-1):
        peak1_idx = price_peaks[i]
        peak2_idx = price_peaks[i+1]

        if peak2_idx - peak1_idx > distance_max:
            continue

        # Regular: price higher high + indicator lower high
        # Hidden: price lower high + indicator higher high
        if (not hidden and price.iloc[peak2_idx] > price.iloc[peak1_idx]) or \
           (hidden and price.iloc[peak2_idx] < price.iloc[peak1_idx]):

            ind_peak1 = ind_peaks[(ind_peaks >= peak1_idx - distance_min) & 
                                (ind_peaks <= peak1_idx + distance_min)]
            ind_peak2 = ind_peaks[(ind_peaks >= peak2_idx - distance_min) & 
                                (ind_peaks <= peak2_idx + distance_min)]

            if len(ind_peak1) > 0 and len(ind_peak2) > 0:
                if (not hidden and indicator.iloc[ind_peak2[0]] < indicator.iloc[ind_peak1[0]]) or \
                   (hidden and indicator.iloc[ind_peak2[0]] > indicator.iloc[ind_peak1[0]]):
                    # For RSI bearish, more significant if peaks are in overbought territory
                    if not is_rsi or indicator.iloc[ind_peak1[0]] > ub:
                        bearish_divs.append((int(peak2_idx), int(ind_peak2[0])))

    bullish_divs = []
    for i in range(len(price_troughs)-1):
        trough1_idx = price_troughs[i]
        trough2_idx = price_troughs[i+1]

        if trough2_idx - trough1_idx > distance_max:
            continue

        # Regular: price lower low + indicator higher low
        # Hidden: price higher low + indicator lower low
        if (not hidden and price.iloc[trough2_idx] < price.iloc[trough1_idx]) or \
           (hidden and price.iloc[trough2_idx] > price.iloc[trough1_idx]):

            ind_trough1 = ind_troughs[(ind_troughs >= trough1_idx - distance_min) & 
                                    (ind_troughs <= trough1_idx + distance_min)]
            ind_trough2 = ind_troughs[(ind_troughs >= trough2_idx - distance_min) & 
                                    (ind_troughs <= trough2_idx + distance_min)]

            if len(ind_trough1) > 0 and len(ind_trough2) > 0:
                if (not hidden and indicator.iloc[ind_trough2[0]] > indicator.iloc[ind_trough1[0]]) or \
                   (hidden and indicator.iloc[ind_trough2[0]] < indicator.iloc[ind_trough1[0]]):
                    # For RSI bullish, more significant if troughs are in oversold territory
                    if not is_rsi or indicator.iloc[ind_trough1[0]] < lb:
                        bullish_divs.append((int(trough2_idx), int(ind_trough2[0])))

    return bearish_divs, bullish_divs


def divergence_signals(df: pd.Series, bearish_divs: list[tuple[int, int]], 
                      bullish_divs: list[tuple[int, int]]) -> pd.Series:

    signal = pd.Series(np.nan, index=df.index)

    for price_idx, _ in bearish_divs:
        signal.iloc[price_idx] = -1

    for price_idx, _ in bullish_divs:
        signal.iloc[price_idx] = 1

    return fill(signal)


def find_double_patterns(hist: pd.Series, distance_min: int = 7, 
                        distance_max: int = 25, prominence: float = 0.05) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:

    prominence *= (hist.max() - hist.min())

    # Find all peaks first
    peaks, _ = find_peaks(hist, 
                          distance=distance_min,
                          prominence=prominence)

    # Filter for positive peaks only
    pos_peaks = peaks[hist.iloc[peaks] > 0]

    # Find all troughs
    troughs, _ = find_peaks(-hist,
                            distance=distance_min,
                            prominence=prominence)

    # Filter for negative troughs only
    neg_troughs = troughs[hist.iloc[troughs] < 0]

    hist = hist.values

    return _find_double_pattern_numba(hist, pos_peaks, neg_troughs, distance_max)


@nb.jit
def _find_double_pattern_numba(hist: np.array, pos_peaks: np.array, neg_troughs: np.array, 
                               distance_max: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:

    double_tops = []
    double_bottoms = []

    # Find double tops (first higher than second)
    for i in range(len(pos_peaks) - 1):
        peak1_idx = pos_peaks[i]
        peak1_val = hist[peak1_idx]

        for j in range(i + 1, len(pos_peaks)):
            peak2_idx = pos_peaks[j]
            peak2_val = hist[peak2_idx]

            if peak2_idx - peak1_idx > distance_max:
                break

            if peak1_val > peak2_val:
                valley = hist[peak1_idx:peak2_idx].min()
                if valley < peak2_val:
                    double_tops.append((int(peak1_idx), int(peak2_idx)))
                    break

    # Find double bottoms (first lower than second)
    for i in range(len(neg_troughs) - 1):
        trough1_idx = neg_troughs[i]
        trough1_val = hist[trough1_idx]

        for j in range(i + 1, len(neg_troughs)):
            trough2_idx = neg_troughs[j]
            trough2_val = hist[trough2_idx]

            if trough2_idx - trough1_idx > distance_max:
                break

            if trough1_val < trough2_val:
                peak = hist[trough1_idx:trough2_idx].max()
                if peak > trough2_val:
                    double_bottoms.append((int(trough1_idx), int(trough2_idx)))
                    break

    return double_tops, double_bottoms


def double_pattern_signals(df: pd.Series, double_tops: list[tuple[int, int]],
                         double_bottoms: list[tuple[int, int]]) -> pd.Series:

    signal = pd.Series(np.nan, index=df.index)

    for _, top2 in double_tops:
        signal.iloc[top2] = -1  # Bearish signal after second peak

    for _, bottom2 in double_bottoms:
        signal.iloc[bottom2] = 1  # Bullish signal after second trough

    return fill(signal)


def vote(signals: pd.DataFrame, threshold: float, weights: list[float]) -> pd.Series:

    weights = np.array(weights)
    combined = signals.dot(weights)
    signal = pd.Series(np.where(combined > threshold, 1, 
                    np.where(combined < -threshold, -1, np.nan)), index=signals.index)
    return fill(signal)


def fill(series: pd.Series, default: int = 1) -> pd.Series:

    if np.isnan(series.iloc[0]):
        series.iloc[0] = default
    return series.ffill().astype(int)

# TODO:
# bb signals: bounce, double, walks, squeeze, breakout, pctB
# better divergence method
# more rsi signals
# ATR, ADX
