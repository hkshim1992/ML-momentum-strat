import numpy as np
import pandas as pd
from src.features import generate_features
from src.model import MLModel


def mac_strategy1(df: pd.DataFrame, sw: int, lw: int, verbose: bool = True):
    """
    Simple moving average crossover strategy.

    Args:
        df (pd.DataFrame): Market data with OHLCV columns.
        sw (int): Short EMA window length.
        lw (int): Long EMA window length.
        verbose (bool): If True, prints final cumulative return.

    Returns:
        pd.DataFrame: Data with signals, positions, and returns.
        float: Final cumulative return of the strategy.
    """
    short_window = sw
    long_window = lw
    data = df.copy()

    # Calculate exponential moving averages for short and long windows
    data['Short_MA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_MA'] = data['Close'].ewm(span=long_window, adjust=False).mean()

    # Remove initial rows to align with long window period
    data = data[long_window:].copy()

    # Generate position signals: 1 if short MA > long MA, else 0
    positions = pd.Series(np.where(data['Short_MA'] > data['Long_MA'], 1, 0), index=data.index)
    signals = positions.diff().fillna(0)  # Identify changes in position (buy/sell signals)

    # Initialize position column with NaN, set buy/sell signals, and forward fill positions
    data['Position'] = np.nan
    data.loc[signals == 1, 'Position'] = 1   # Buy signal
    data.loc[signals == -1, 'Position'] = 0  # Sell signal
    data['Position'] = data['Position'].ffill().fillna(0)  # Carry forward positions and fill initial NaNs

    data['Signal'] = data['Position'].diff().fillna(0)  # Signal changes for visualization

    # Capture buy and sell prices for potential plotting/analysis
    data['Buy_Price'] = np.where(data['Signal'] == 1, data['Close'], np.nan)
    data['Sell_Price'] = np.where(data['Signal'] == -1, data['Close'], np.nan)

    # Calculate daily returns only when in position
    data['Daily_Return'] = np.where(data['Position'].shift() == 1, data['Close'].pct_change(), 0)
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()  # Compound returns

    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1  # Total return over period

    if verbose:
        print(f'Final cumulative return of the strategy: {100 * final_cum_return:.2f}%')

    return data, final_cum_return


def mac_strategy_ml(
    df: pd.DataFrame,
    sw: int,
    lw: int,
    ml_model: MLModel,
    ml_threshold: float = 0.7,
    verbose: bool = True,
):
    """
    Moving average crossover strategy enhanced with ML signal integration.

    ML signals can override traditional rule-based signals based on confidence threshold.

    Args:
        df (pd.DataFrame): Raw market data with OHLCV columns.
        sw (int): Short EMA window length.
        lw (int): Long EMA window length.
        ml_model (MLModel): Trained ML model instance.
        ml_threshold (float): Probability threshold for accepting ML signal.
        verbose (bool): If True, prints final cumulative return.

    Returns:
        pd.DataFrame: Data with combined signals, positions, and returns.
        float: Final cumulative return of the strategy.
    """
    if not 0.5 <= ml_threshold < 1.0:
        raise ValueError("ml_threshold must be in [0.5, 1.0).")

    # Step 1: Generate features and get ML predictions.
    # Do not create future-dependent labels for inference-time signals.
    df_ml = generate_features(df, include_target=False)
    if df_ml.empty:
        raise ValueError("No feature rows available for ML strategy.")

    # Predict probability for positive class
    ml_probs = ml_model.predict_proba(df_ml)
    ml_probs_series = pd.Series(ml_probs, index=df_ml.index, name="ML_Prob_Up")

    # Initialize ML signals series with default -1 (indicating no/low confidence)
    ml_signals = pd.Series(-1, index=df_ml.index)

    # Symmetric confidence band:
    #   buy  -> p(up) >= t
    #   sell -> p(up) <= (1 - t)
    lower_threshold = 1.0 - ml_threshold
    ml_signals.loc[ml_probs_series >= ml_threshold] = 1
    ml_signals.loc[ml_probs_series <= lower_threshold] = 0

    # Step 2: Compute rule-based EMA crossover signals
    df_rule_base = df.copy()
    df_rule_base['Short_MA'] = df_rule_base['Close'].ewm(span=sw, adjust=False).mean()
    df_rule_base['Long_MA'] = df_rule_base['Close'].ewm(span=lw, adjust=False).mean()
    df_rule_base = df_rule_base[lw:].copy()  # Align index with long window period

    rule_signals = (df_rule_base['Short_MA'] > df_rule_base['Long_MA']).astype(int)

    # Step 3: Align ML signals with rule-based signals dataframe
    ml_signals = ml_signals.reindex(df_rule_base.index, fill_value=-1)

    # Step 4: Combine ML and rule-based signals
    # ML signal overrides rule signal if confident; otherwise use rule signal
    data = df_rule_base.copy()
    data['ML_Prob_Up'] = ml_probs_series.reindex(df_rule_base.index)
    data['ML_Signal'] = ml_signals
    data['Position'] = np.where(
        ml_signals == 1, 1,
        np.where(ml_signals == 0, 0, rule_signals)
    )

    # Step 5: Generate trade signals and positions for backtesting
    signals = data['Position'].diff().fillna(0)

    data['Position'] = np.nan
    data.loc[signals == 1, 'Position'] = 1   # Buy signal
    data.loc[signals == -1, 'Position'] = 0  # Sell signal
    data['Position'] = data['Position'].ffill().fillna(0)  # Forward fill positions, fill NaNs with 0

    data['Signal'] = data['Position'].diff().fillna(0)

    # Step 6: Calculate returns
    data['Buy_Price'] = np.where(data['Signal'] == 1, data['Close'], np.nan)
    data['Sell_Price'] = np.where(data['Signal'] == -1, data['Close'], np.nan)
    data['Daily_Return'] = np.where(data['Position'].shift() == 1, data['Close'].pct_change(), 0)
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()

    final_return = data['Cumulative_Return'].iloc[-1] - 1

    if verbose:
        print(f'Final cumulative return of the strategy with ML: {100 * final_return:.2f}%')

    return data, final_return
