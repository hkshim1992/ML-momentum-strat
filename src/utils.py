import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

def plot_strategy(data: pd.DataFrame, ticker: str):
    """
    Plot the price, moving averages, buy/sell signals, and position over time.

    Args:
        data (pd.DataFrame): Data containing Close price, MAs, signals, and positions.
        ticker (str): Ticker symbol for the title.
    """
    _, ax = plt.subplots(2, 1, sharex=True, height_ratios=(8, 2), figsize=(10, 8))

    # Plot close price and moving averages
    data['Close'].plot(ax=ax[0], label='Close')
    data['Short_MA'].plot(ax=ax[0], label='Short MA', linewidth=1)
    data['Long_MA'].plot(ax=ax[0], label='Long MA', linewidth=1)

    # Plot buy and sell markers
    data['Buy_Price'].plot(ax=ax[0], label='Buy', marker='^', color='b', markersize=8)
    data['Sell_Price'].plot(ax=ax[0], label='Sell', marker='v', color='r', markersize=8)

    ax[0].set_title(f'{ticker} Trades', fontsize=18)
    ax[0].set_ylabel('Price ($)', fontsize=12)
    ax[0].legend(fontsize=12)
    ax[0].grid(alpha=0.3)

    # Plot position (1 or 0)
    data['Position'].plot(ax=ax[1])
    ax[1].set_xlabel('Date', fontsize=12)
    ax[1].set_ylabel('Position', fontsize=12)
    ax[1].grid(alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_cumulative_returns(strategies: dict, ticker: str = ''):
    """
    Plot cumulative returns over time for multiple strategies.

    Args:
        strategies (dict): Keys are strategy names, values are dataframe series with cumulative return.
        ticker (str): Optional. Ticker symbol to include in plot title.
    """
    colors = ['red', 'blue', 'black']
    linestyles = ['-', '-', '--']
    plt.figure(figsize=(12, 6))

    for i, (name, series) in enumerate(strategies.items()):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(series.index, series, label=name, color=color, linestyle=linestyle)

    plt.title(f'Cumulative Returns Comparison {f'for {ticker}' if ticker else ''}', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def tear_sheet1(data: pd.DataFrame):
    """
    Print a summary performance tear sheet for the strategy.

    Includes:
      - Trading period length
      - Final cumulative return vs buy & hold
      - CAGR of strategy and benchmark
      - Sharpe ratio
      - Maximum Drawdown (MDD)
      - Win rate, average holding period
      - Average profit/loss per trade and profit/loss ratio

    Args:
        data (pd.DataFrame): Data with columns including 'Close', 'Cumulative_Return', 'Signal'.
    """
    trading_days_per_year = 252
    trading_period = len(data) / trading_days_per_year  # in years
    print(f'Trading Period: {trading_period:.1f} years')

    # Final returns: strategy vs buy & hold
    buy_and_hold = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1
    final_cum_return = data['Cumulative_Return'].iloc[-1] - 1
    print(f'Final cumulative return of the strategy: {100*final_cum_return:.2f}%, Buy & Hold: {100*buy_and_hold:.2f}%')

    # CAGR calculation
    CAGR_strategy = (data['Cumulative_Return'].iloc[-1])**(1/trading_period) - 1
    CAGR_benchmark = (buy_and_hold + 1)**(1/trading_period) - 1
    print(f'Strategy CAGR: {100*CAGR_strategy:.2f}%, Benchmark CAGR: {100*CAGR_benchmark:.2f}%')

    # Sharpe Ratio (annualized, risk-free rate assumed 0.3%)
    risk_free_rate = 0.003
    strategy_daily_return = data['Cumulative_Return'].pct_change().fillna(0)
    mean_return = strategy_daily_return.mean() * trading_days_per_year
    std_return = strategy_daily_return.std() * np.sqrt(trading_days_per_year)
    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    # Maximum Drawdown (MDD) for strategy
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1
    max_drawdown = data['Drawdown'].min()

    # Benchmark MDD based on buy & hold returns
    cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    mdd_benchmark = drawdown.min()

    print(f'Strategy MDD: {100*max_drawdown:.2f}%, Benchmark MDD: {100*mdd_benchmark:.2f}%')

    # Win rate and average holding period calculations
    buy_signals = data[data['Signal'] == 1].index
    sell_signals = data[data['Signal'] == -1].index

    returns = []
    holding_periods = []

    for buy_date in buy_signals:
        sell_dates = sell_signals[sell_signals > buy_date]
        if not sell_dates.empty:
            sell_date = sell_dates[0]
            buy_price = data.loc[buy_date, 'Close']
            sell_price = data.loc[sell_date, 'Close']
            return_pct = sell_price / buy_price - 1
            returns.append(return_pct)
            holding_period = np.busday_count(buy_date.date(), sell_date.date())
            holding_periods.append(holding_period)
    
    # Handle edge case: if last buy has no matching sell, close at last available price
    if len(buy_signals) > len(sell_signals):
        last_buy = buy_signals[-1]
        last_date = data.index[-1]
        if last_date > last_buy:
            buy_price = data.loc[last_buy, 'Close']
            sell_price = data.loc[last_date, 'Close']
            return_pct = sell_price / buy_price - 1
            returns.append(return_pct)
            holding_period = np.busday_count(last_buy.date(), last_date.date())
            holding_periods.append(holding_period)
            print(f"Warning: Last buy on {last_buy.date()} closed at last date {last_date.date()}")

    profitable_trades = len([r for r in returns if r > 0])
    loss_trades = len([r for r in returns if r <= 0])
    total_trades = len(returns)
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    print(f'Number of Profitable Trades: {profitable_trades}, Number of Loss Trades: {loss_trades}, Win Rate: {100*win_rate:.2f}%')

    average_holding_period = np.mean(holding_periods) if holding_periods else 0
    print(f'Average Holding Period: {average_holding_period:.1f} days')

    average_profit = np.mean([r for r in returns if r > 0]) if profitable_trades > 0 else 0
    average_loss = np.mean([r for r in returns if r <= 0]) if loss_trades > 0 else 0
    print(f'Avg ROR/trade in profitable trades: {100*average_profit:.3f}%, Avg ROR/trade in loss trades: {100*average_loss:.3f}%')

    if average_loss == 0:
        print("Profit/Loss Ratio: Infinite (no losing trades)")
    else:
        profit_loss_ratio = average_profit / abs(average_loss)
        print(f'Profit/Loss Ratio: {profit_loss_ratio:.2f}')
    # profit_loss_ratio = average_profit / abs(average_loss) if average_loss != 0 else np.inf
    # print(f'Profit/Loss Ratio: {profit_loss_ratio:.2f}')


def rolling_test(ticker: str, date: str, strat, opt, **kwargs):
    """
    Run rolling-window backtest with parameter optimization on training set and test on subsequent data.

    Args:
        ticker (str): Stock ticker symbol.
        date (str): Rolling window start date (YYYY-MM-DD).
        strat (function): Strategy function to backtest.
        opt (function): Parameter optimizer function.
        **kwargs: Additional args for strat/opt.

    Returns:
        tuple: CAGR_strategy, MDD_strategy, CAGR_benchmark, MDD_benchmark.
    """
    middle_date = date
    middle_date_dt = datetime.strptime(middle_date, '%Y-%m-%d')
    start_date_dt = middle_date_dt.replace(year=middle_date_dt.year - 5)
    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date_dt = middle_date_dt.replace(year=middle_date_dt.year + 2)
    end_date = end_date_dt.strftime('%Y-%m-%d')

    # Download data for rolling window period
    df = yf.download(ticker, start_date, end_date)
    df.columns = df.columns.get_level_values(0)  # flatten multiindex if present

    # Split data into training (up to middle_date) and testing (after)
    df_train = df.loc[start_date:middle_date].copy()
    optimal_params, optimal_df, param_results_df = opt(df_train, strat, **kwargs)
    df_test = df.loc[middle_date:].copy()
    data, ret = strat(df_test, *optimal_params[:2], **kwargs)

    # Calculate CAGR
    fee_rate = 0.001
    trading_period = len(data) / 252  # number of trading days in a year
    buy_and_hold = data['Close'].iloc[-1] * (1 - fee_rate) / (data['Close'].iloc[0] * (1 + fee_rate))
    CAGR_strategy = (data['Cumulative_Return'].iloc[-1]) ** (1 / trading_period) - 1
    CAGR_benchmark = buy_and_hold ** (1 / trading_period) - 1

    # Calculate max drawdown for strategy
    data['Cumulative_Max'] = data['Cumulative_Return'].cummax()
    data['Drawdown'] = data['Cumulative_Return'] / data['Cumulative_Max'] - 1
    mdd_strategy = data['Drawdown'].min()

    # Calculate max drawdown for benchmark
    cumulative_returns = (1 + data['Close'].pct_change()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    mdd_benchmark = drawdown.min()

    return CAGR_strategy, mdd_strategy, CAGR_benchmark, mdd_benchmark


def plot_rolling_performance(results_df: pd.DataFrame, dates: list[str], ticker: str):
    """
    Plot rolling-window CAGR and Maximum Drawdown (MDD) comparisons for strategy and benchmark.

    Args:
        results_df (pd.DataFrame): Multi-index DataFrame with ('Strategy','CAGR'), ('Benchmark','CAGR'), etc.
        dates (list[str]): List of date strings for x-axis labels.
        ticker (str): Stock ticker symbol for title.
    """
    # Extract values for plotting
    values1 = results_df[('Strategy', 'CAGR')].values
    values2 = results_df[('Benchmark', 'CAGR')].values
    values3 = results_df[('Strategy', 'MDD')].values
    values4 = results_df[('Benchmark', 'MDD')].values

    bar_width = 0.3
    index = np.arange(len(dates))

    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot CAGR bars
    ax[0].bar(index, values1 * 100, bar_width, label='CAGR Strategy')
    ax[0].bar(index + bar_width, values2 * 100, bar_width, label='CAGR Benchmark')

    # Plot MDD bars
    ax[1].bar(index, values3 * 100, bar_width, label='MDD Strategy')
    ax[1].bar(index + bar_width, values4 * 100, bar_width, label='MDD Benchmark')

    # Axis labels and titles
    ax[0].set_ylabel('CAGR (%)', fontsize=15)
    ax[0].set_title(f'MAC Rolling CAGR and MDD for {ticker}', fontsize=20)

    ax[1].set_ylabel('MDD (%)', fontsize=15)
    ax[1].set_xlabel('Dates', fontsize=15)
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels(dates)

    ax[0].legend(fontsize=13)
    ax[1].legend(fontsize=13)

    ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
