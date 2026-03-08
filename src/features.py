import pandas as pd

try:
    import ta
except ModuleNotFoundError:
    ta = None


def _rsi_fallback(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd_fallback(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    return ema_fast - ema_slow

def generate_features(
    df: pd.DataFrame,
    forward_days: int = 5,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Generate technical indicators and features for ML modeling.
    
    Parameters:
        df (pd.DataFrame): Raw price and volume data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns:
        pd.DataFrame: Original data with additional feature columns and, optionally, target label.
    """
    df = df.copy()
    
    # Technical indicators
    period = 14
    if ta is None:
        df['RSI'] = _rsi_fallback(df['Close'], window=period)
        df['MACD'] = _macd_fallback(df['Close'])
    else:
        df['RSI'] = ta.momentum.rsi(df.Close, window=period)
        df['MACD'] = ta.trend.macd(df.Close)
    # bb = ta.volatility.BollingerBands(close=df['Close'])
    # df['bb_bbm'] = bb.bollinger_mavg()
    # df['bb_bbh'] = bb.bollinger_hband()
    # df['bb_bbl'] = bb.bollinger_lband()
    # df['bb_width'] = bb.bollinger_wband()

    # Lagged returns
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)

    # Moving averages and momentum
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['mom_10d'] = df['Close'] - df['Close'].shift(10)

    # Volatility
    df['vola_5d'] = df['Close'].rolling(5).std()

    # Volume-based features
    df['vol_mean5'] = df['Volume'].rolling(5).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_mean5']

    # Future label (optional for train/eval only).
    if include_target:
        df['fwd_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
        df['target'] = (df['fwd_return'] > 0).astype(int)

    # Drop rows where engineered inputs are unavailable.
    required_for_model = [
        'RSI',
        'MACD',
        'return_1d',
        'return_3d',
        'return_5d',
        'ma5',
        'ma10',
        'mom_10d',
        'vola_5d',
        'vol_mean5',
        'vol_ratio',
    ]
    if include_target:
        required_for_model.append('fwd_return')
    df.dropna(subset=required_for_model, inplace=True)

    return df
