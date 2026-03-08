import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import MLModel
from src.utils import plot_cumulative_returns


def _sample_training_df(n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100, 120, n)
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.linspace(1000, 3000, n),
            "feat_1": np.sin(np.linspace(0, 3, n)),
            "feat_2": np.cos(np.linspace(0, 3, n)),
        },
        index=idx,
    )
    # Ensure both classes exist.
    df["target"] = ([0, 1] * (n // 2)) + ([0] if n % 2 else [])
    return df


def test_predict_before_train_raises_not_fitted():
    model = MLModel()
    with pytest.raises(NotFittedError):
        model.predict(_sample_training_df(10))


def test_predict_does_not_refit_scaler(monkeypatch):
    train_df = _sample_training_df(50)
    pred_df = _sample_training_df(20).drop(columns=["target"])

    model = MLModel()
    model.preprocess_data(train_df, fit=True)
    mean_before = model.scaler.mean_.copy()

    monkeypatch.setattr(model.model, "predict", lambda X: np.zeros(len(X), dtype=int))
    preds = model.predict(pred_df)
    mean_after = model.scaler.mean_.copy()

    assert len(preds) == len(pred_df)
    assert np.array_equal(mean_before, mean_after)


def test_plot_cumulative_returns_runs_without_error(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    strategies = {
        "Strategy": pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx),
        "Benchmark": pd.Series([1.0, 0.99, 1.0, 1.01, 1.02], index=idx),
    }
    plot_cumulative_returns(strategies, ticker="AAPL")
