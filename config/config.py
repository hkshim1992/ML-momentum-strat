# Window sizes used across strategies for moving averages
# short_window: list of possible short-term moving average spans to test
# long_window: list of possible long-term moving average spans to test
shared_windows = {
    'short_window': list(range(5, 22)),  # Short MA windows from 5 to 21 days
    'long_window': list(range(22, 43))   # Long MA windows from 22 to 42 days
}