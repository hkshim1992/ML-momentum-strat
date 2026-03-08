import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.features import generate_features
from src.model import MLModel
from src.strategy import mac_strategy1, mac_strategy_ml


REQUIRED_OHLCV_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def _normalize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing_cols = REQUIRED_OHLCV_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    return df.sort_index().copy()


def _load_market_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv is not None:
        df = pd.read_csv(args.csv)
        if args.date_column not in df.columns:
            raise ValueError(
                f"CSV is missing date column '{args.date_column}'. "
                "Use --date-column to override."
            )
        df[args.date_column] = pd.to_datetime(df[args.date_column])
        df = df.set_index(args.date_column)
        return _normalize_market_data(df)

    if not args.ticker:
        raise ValueError("Ticker is required when --csv is not provided.")
    if not args.start or not args.end:
        raise ValueError("Start/end dates are required when --csv is not provided.")

    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yfinance is required for ticker-based downloads. "
            "Install dependencies or provide --csv instead."
        ) from exc

    df = yf.download(
        args.ticker,
        start=args.start,
        end=args.end,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError("No rows were downloaded for the given ticker/date range.")
    return _normalize_market_data(df)


def _add_data_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path, default=None, help="Path to OHLCV CSV input.")
    parser.add_argument(
        "--date-column",
        type=str,
        default="Date",
        help="Date column name for --csv input.",
    )
    parser.add_argument("--ticker", type=str, default=None, help="Ticker for Yahoo Finance download.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD).")


def _load_model_payload(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        payload = {"model": payload, "metadata": {}, "metrics": {}}
    model = payload.get("model")
    if not isinstance(model, MLModel):
        raise TypeError("Saved model payload does not contain MLModel.")
    payload.setdefault("metadata", {})
    payload.setdefault("metrics", {})
    return payload


def _learn_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.5, 0.951, 0.01)
    best_threshold = 0.5
    best_score = -np.inf

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
    return best_threshold, float(best_score)


def _fit_model_with_threshold(
    df_labeled: pd.DataFrame,
    model_type: str,
    threshold_mode: str,
    fixed_threshold: float,
    val_ratio: float,
) -> tuple[MLModel, float, dict[str, Any]]:
    if threshold_mode not in {"auto", "fixed"}:
        raise ValueError("threshold-mode must be one of: auto, fixed.")
    if not 0.5 <= fixed_threshold < 1.0:
        raise ValueError("Fixed threshold must be in [0.5, 1.0).")
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio must be in (0, 1).")

    # Fixed threshold path.
    if threshold_mode == "fixed":
        model = MLModel(model_type=model_type)
        model.train(df_labeled)
        return model, fixed_threshold, {"threshold_source": "fixed"}

    # Auto threshold path with chronological validation split.
    val_size = max(1, int(len(df_labeled) * val_ratio))
    if val_size >= len(df_labeled):
        raise ValueError("val-ratio leaves no rows for training.")

    train_sel = df_labeled.iloc[:-val_size].copy()
    val_df = df_labeled.iloc[-val_size:].copy()
    if train_sel.empty or val_df.empty:
        raise ValueError("Unable to form non-empty train/validation splits for auto threshold.")

    model_sel = MLModel(model_type=model_type)
    try:
        model_sel.train(train_sel)
    except ValueError as exc:
        if "at least two classes" not in str(exc):
            raise
        model = MLModel(model_type=model_type)
        model.train(df_labeled)
        return model, fixed_threshold, {
            "threshold_source": "fixed_fallback_single_class_train_split",
            "validation_rows": int(len(val_df)),
            "validation_balanced_accuracy": None,
        }

    y_val = val_df["target"].to_numpy()
    if len(np.unique(y_val)) < 2:
        selected_threshold = fixed_threshold
        threshold_info = {
            "threshold_source": "fixed_fallback_single_class_validation",
            "validation_rows": int(len(val_df)),
            "validation_balanced_accuracy": None,
        }
    else:
        val_probs = model_sel.predict_proba(val_df)
        selected_threshold, best_bal_acc = _learn_best_threshold(y_val, val_probs)
        threshold_info = {
            "threshold_source": "auto_validation",
            "validation_rows": int(len(val_df)),
            "validation_balanced_accuracy": best_bal_acc,
        }

    # Refit final model on full in-sample labeled data after threshold selection.
    model = MLModel(model_type=model_type)
    model.train(df_labeled)
    return model, float(selected_threshold), threshold_info


def cmd_train(args: argparse.Namespace) -> int:
    df = _load_market_data(args)
    df_features = generate_features(df, include_target=True)
    if len(df_features) < 20:
        raise ValueError("Not enough feature rows after preprocessing; need at least 20.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be in (0, 1).")

    split_idx = int(len(df_features) * args.train_ratio)
    if split_idx <= 0 or split_idx >= len(df_features):
        raise ValueError("train-ratio yields an empty train or test split.")

    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()

    model, selected_threshold, threshold_info = _fit_model_with_threshold(
        train_df,
        model_type=args.model_type,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.prob_threshold,
        val_ratio=args.val_ratio,
    )

    test_probs = model.predict_proba(test_df)
    test_pred = (test_probs >= selected_threshold).astype(int)
    test_true = test_df["target"].to_numpy()

    metrics: dict[str, Any] = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "accuracy": float(accuracy_score(test_true, test_pred)),
        "selected_threshold": float(selected_threshold),
        "threshold_mode": args.threshold_mode,
        "threshold_source": threshold_info["threshold_source"],
    }
    if len(np.unique(test_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(test_true, test_probs))
    else:
        metrics["roc_auc"] = None
    if "validation_balanced_accuracy" in threshold_info:
        metrics["validation_balanced_accuracy"] = threshold_info["validation_balanced_accuracy"]

    payload = {
        "model": model,
        "metadata": {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_type": args.model_type,
            "train_ratio": args.train_ratio,
            "prob_threshold": args.prob_threshold,
            "selected_threshold": float(selected_threshold),
            "threshold_mode": args.threshold_mode,
            "threshold_source": threshold_info["threshold_source"],
            "val_ratio": args.val_ratio,
            "source": "csv" if args.csv else "yfinance",
            "ticker": args.ticker,
            "start": args.start,
            "end": args.end,
            "feature_columns": model.feature_cols_,
        },
        "metrics": metrics,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as fh:
        pickle.dump(payload, fh)

    print(f"Saved model to: {args.model_out}")
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    payload = _load_model_payload(args.model_in)
    model: MLModel = payload["model"]
    metadata = payload.get("metadata", {})
    df = _load_market_data(args)
    df_features = generate_features(df, include_target=False)
    if df_features.empty:
        raise ValueError("No rows available for prediction after feature generation.")

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(metadata.get("selected_threshold", 0.5))
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must be in [0.5, 1.0).")

    probs = model.predict_proba(df_features)
    preds = (probs >= threshold).astype(int)
    lower_threshold = 1.0 - threshold
    signals = np.where(probs >= threshold, 1, np.where(probs <= lower_threshold, 0, -1))
    out_df = pd.DataFrame(
        {
            "open": df_features["Open"],
            "high": df_features["High"],
            "low": df_features["Low"],
            "close": df_features["Close"],
            "volume": df_features["Volume"],
            "prediction": preds,
            "prob_up": probs,
            "signal": signals,
            "threshold_used": threshold,
        },
        index=df_features.index,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index_label="Date")

    print(f"Saved predictions to: {args.output}")
    print(out_df.tail(args.last_n).to_string())
    return 0


def _run_strategy_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    df = _load_market_data(args)
    if args.long_window <= args.short_window:
        raise ValueError("long-window must be greater than short-window.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio must be in (0, 1).")

    if args.strategy == "mac":
        data, total_return = mac_strategy1(df, args.short_window, args.long_window, verbose=False)
        return data, total_return, {"selected_threshold": None, "threshold_source": None, "oos_start": str(data.index[0])}

    # Strict out-of-sample setup for ML strategy.
    df_features = generate_features(df, include_target=True)
    split_idx = int(len(df_features) * args.train_ratio)
    if split_idx <= 0 or split_idx >= len(df_features):
        raise ValueError("train-ratio yields an empty train or test split.")

    train_df = df_features.iloc[:split_idx].copy()
    test_labeled_df = df_features.iloc[split_idx:].copy()
    if test_labeled_df.empty:
        raise ValueError("No out-of-sample rows for evaluation after split.")
    test_start = test_labeled_df.index[0]

    model, selected_threshold, threshold_info = _fit_model_with_threshold(
        train_df,
        model_type="logistic",
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.ml_threshold,
        val_ratio=args.val_ratio,
    )

    # Include long-window warmup before OOS boundary to initialize indicators/MAs,
    # then slice to strict OOS for reporting.
    start_pos = df.index.get_loc(test_start)
    if isinstance(start_pos, slice):
        start_pos = start_pos.start
    warmup_start = max(0, int(start_pos) - args.long_window)
    df_for_strategy = df.iloc[warmup_start:].copy()

    data, total_return = mac_strategy_ml(
        df_for_strategy,
        args.short_window,
        args.long_window,
        ml_model=model,
        ml_threshold=selected_threshold,
        verbose=False,
    )
    data = data.loc[test_start:].copy()
    if data.empty:
        raise ValueError("No OOS rows remain after strategy warmup.")

    # Rebase cumulative return to OOS boundary.
    oos_base = data["Cumulative_Return"].iloc[0]
    if oos_base == 0:
        raise ValueError("Unexpected zero cumulative base during OOS rebasing.")
    data["Cumulative_Return"] = data["Cumulative_Return"] / oos_base
    total_return = float(data["Cumulative_Return"].iloc[-1] - 1)

    return data, total_return, {
        "selected_threshold": float(selected_threshold),
        "threshold_source": threshold_info["threshold_source"],
        "oos_start": str(test_start),
    }


def cmd_backtest(args: argparse.Namespace) -> int:
    data, total_return, backtest_meta = _run_strategy_backtest(args)
    buy_hold_return = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1

    summary = {
        "strategy": args.strategy,
        "rows": int(len(data)),
        "return_strategy": float(total_return),
        "return_buy_hold": float(buy_hold_return),
        "selected_threshold": backtest_meta.get("selected_threshold"),
        "threshold_source": backtest_meta.get("threshold_source"),
        "oos_start": backtest_meta.get("oos_start"),
    }
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index_label="Date")
        print(f"Saved backtest rows to: {args.output}")

    ticker_label = args.ticker or "CSV"
    if args.plot_strategy:
        from src.utils import plot_strategy

        plot_strategy(data, ticker=ticker_label)
    if args.plot_cumulative:
        from src.utils import plot_cumulative_returns

        strategies = {"Strategy": data["Cumulative_Return"]}
        benchmark = (1 + data["Close"].pct_change().fillna(0)).cumprod()
        strategies["Buy & Hold"] = benchmark
        plot_cumulative_returns(strategies, ticker=ticker_label)
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    data = pd.read_csv(args.results_csv, parse_dates=[args.date_column])
    data = data.set_index(args.date_column).sort_index()

    ticker_label = args.ticker_label or "Results"
    if args.mode in {"strategy", "both"}:
        from src.utils import plot_strategy

        required = {"Close", "Short_MA", "Long_MA", "Buy_Price", "Sell_Price", "Position"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Strategy plot requires columns: {sorted(required)}; missing {sorted(missing)}")
        plot_strategy(data, ticker=ticker_label)

    if args.mode in {"cumulative", "both"}:
        from src.utils import plot_cumulative_returns

        if "Cumulative_Return" not in data.columns:
            raise ValueError("Cumulative plot requires 'Cumulative_Return' column.")
        series_map = {"Strategy": data["Cumulative_Return"]}
        if "Close" in data.columns:
            series_map["Buy & Hold"] = (1 + data["Close"].pct_change().fillna(0)).cumprod()
        plot_cumulative_returns(series_map, ticker=ticker_label)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-momentum-cli",
        description="CLI for training, prediction, backtesting, and visualization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train ML model and persist artifact.")
    _add_data_source_args(train)
    train.add_argument("--model-type", type=str, default="logistic", choices=["logistic"])
    train.add_argument("--train-ratio", type=float, default=0.7)
    train.add_argument("--val-ratio", type=float, default=0.2)
    train.add_argument("--threshold-mode", type=str, choices=["auto", "fixed"], default="auto")
    train.add_argument("--prob-threshold", type=float, default=0.5)
    train.add_argument("--model-out", type=Path, default=Path("artifacts/model.pkl"))
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Load model artifact and generate predictions.")
    _add_data_source_args(predict)
    predict.add_argument("--model-in", type=Path, required=True)
    predict.add_argument("--threshold", type=float, default=None)
    predict.add_argument("--last-n", type=int, default=10)
    predict.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    predict.set_defaults(func=cmd_predict)

    backtest = sub.add_parser("backtest", help="Run rule-based or ML-augmented strategy backtest.")
    _add_data_source_args(backtest)
    backtest.add_argument("--strategy", choices=["mac", "mac-ml"], default="mac")
    backtest.add_argument("--short-window", type=int, default=10)
    backtest.add_argument("--long-window", type=int, default=30)
    backtest.add_argument("--train-ratio", type=float, default=0.7)
    backtest.add_argument("--val-ratio", type=float, default=0.2)
    backtest.add_argument("--threshold-mode", type=str, choices=["auto", "fixed"], default="auto")
    backtest.add_argument("--ml-threshold", type=float, default=0.7)
    backtest.add_argument("--plot-strategy", action="store_true")
    backtest.add_argument("--plot-cumulative", action="store_true")
    backtest.add_argument("--output", type=Path, default=Path("artifacts/backtest.csv"))
    backtest.set_defaults(func=cmd_backtest)

    vis = sub.add_parser("visualize", help="Visualize saved backtest/results CSV.")
    vis.add_argument("--results-csv", type=Path, required=True)
    vis.add_argument("--date-column", type=str, default="Date")
    vis.add_argument("--ticker-label", type=str, default=None)
    vis.add_argument("--mode", choices=["strategy", "cumulative", "both"], default="both")
    vis.set_defaults(func=cmd_visualize)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
