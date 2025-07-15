import pandas as pd
import ta

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators and features for ML modeling.
    
    Parameters:
        df (pd.DataFrame): Raw price and volume data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns:
        pd.DataFrame: Original data with additional feature columns and target label.
    """
    df = df.copy()
    
    # Technical indicators
    period = 14
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

    # Forward return and label
    forward_days = 5
    df['fwd_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
    df['target'] = (df['fwd_return'] > 0).astype(int)

    # drop NAs
    df.dropna(inplace=True)

    return df
