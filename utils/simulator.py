import matplotlib.pyplot as plt

def simulate_spike_trading(prices, spike_indices):
    """
    Simulates a simple buy-on-spike, sell-on-next-spike strategy. It really don't matter if the spike is up or down, it is intentionally super dumb.
    Assumes full buy/sell at spike points. No transaction fees/slippage.

    Args:
        prices (array-like): Price series
        spike_indices (list): Indices where spikes occurred (binary 1s)

    Returns:
        trades: List of dicts with entry, exit, profit
        cumulative_profit: List of profit values over time
    """
    position = None
    buy_price = 0
    trades = []
    cumulative_profit = []
    profit = 0

    for i in range(len(prices)):
        if i in spike_indices:
            if position is None:
                # BUY
                buy_price = prices[i]
                position = i
            else:
                # SELL
                sell_price = prices[i]
                trade_profit = sell_price - buy_price
                profit += trade_profit
                trades.append({
                    "entry_day": position,
                    "exit_day": i,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "profit": trade_profit
                })
                position = None  # reset

        cumulative_profit.append(float(profit))

    return trades, cumulative_profit
