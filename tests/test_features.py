import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import generate_features


def _make_ohlcv(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=rows, freq="B")
    close = 100 + np.sin(np.linspace(0, 16, rows)) * 4 + np.linspace(0, 10, rows)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1000, 2000, rows),
        },
        index=idx,
    )


def test_generate_features_prediction_mode_keeps_latest_row():
    raw = _make_ohlcv()
    feat = generate_features(raw, include_target=False)

    assert "target" not in feat.columns
    assert "fwd_return" not in feat.columns
    assert feat.index.max() == raw.index.max()


def test_generate_features_train_mode_has_target_and_drops_forward_tail():
    raw = _make_ohlcv()
    feat = generate_features(raw, include_target=True, forward_days=5)

    assert "target" in feat.columns
    assert "fwd_return" in feat.columns
    assert feat.index.max() < raw.index.max()
