import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main


def _write_ohlcv_csv(path: Path, rows: int = 120) -> None:
    idx = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = np.linspace(100, 150, rows)
    df = pd.DataFrame(
        {
            "Date": idx,
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1000, 5000, rows),
        }
    )
    df.to_csv(path, index=False)


def test_backtest_mac_from_csv_writes_output(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    out_path = tmp_path / "backtest.csv"
    _write_ohlcv_csv(csv_path)

    rc = main(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--short-window",
            "10",
            "--long-window",
            "30",
            "--strategy",
            "mac",
            "--output",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()
    out_df = pd.read_csv(out_path)
    assert {"Close", "Cumulative_Return", "Position", "Signal"}.issubset(set(out_df.columns))


def test_visualize_cumulative_mode_smoke(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "sample.csv"
    out_path = tmp_path / "backtest.csv"
    _write_ohlcv_csv(csv_path)
    main(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--short-window",
            "10",
            "--long-window",
            "30",
            "--strategy",
            "mac",
            "--output",
            str(out_path),
        ]
    )

    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    rc = main(
        [
            "visualize",
            "--results-csv",
            str(out_path),
            "--mode",
            "cumulative",
        ]
    )
    assert rc == 0
