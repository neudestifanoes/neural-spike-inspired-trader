import yfinance as yf
from Entropy.entropy_core import calculate_entropy

def analyze_single_stock(ticker, threshold, window_size, start_date, end_date):
    """
    Downloads data, computes spike statistics and entropy for a single stock.
    Returns: dict with spike data, entropy data, and summary stats.
    """
    # Download data (pandas treats dates as the index, and we are using the closing prices for prices)
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    prices = data["Close"].values
    dates = data.index[1:]

    # Price change → spike detection
    pct_changes = (prices[1:] - prices[:-1]) / prices[:-1] #creates a list of everyday percent changes back to back
    up_indices = [i for i, change in enumerate(pct_changes) if change >= threshold] # if any of the numbers calculated above are greater than threshold, defined in main, then it returns that index 
    down_indices = [i for i, change in enumerate(pct_changes) if change <= -threshold]

    # Dates and prices for spikes
    up_dates = [dates[i] for i in up_indices]
    down_dates = [dates[i] for i in down_indices]
    up_prices = [prices[i + 1] for i in up_indices]
    down_prices = [prices[i + 1] for i in down_indices]

    # Binary spike train 
    binary_spikes = [1 if i in up_indices or i in down_indices else 0 for i in range(len(dates))]  #creates that list of 0, 1, 1, 0, it is actually a numpy array, it marks each day as 1 or 0

    # Entropy windowing
    entropy_values = []
    entropy_dates = []
    for i in range(0, len(binary_spikes) - window_size + 1): #window size is a chunk of days that will be defined in main, helps us calculate entropy for multiple windows and see the regularity and changes of the spikes
        window = binary_spikes[i:i + window_size]
        entropy = calculate_entropy(window)
        entropy_values.append(entropy)
        entropy_dates.append(dates[i + window_size // 2])

    # Summary stats
    total_days = len(prices)
    up_rate = len(up_indices) / total_days * 100
    down_rate = len(down_indices) / total_days * 100

    return {
        "data": data,
        "prices": prices,
        "dates": dates,
        "up_spike_dates": up_dates,
        "up_spike_prices": up_prices,
        "down_spike_dates": down_dates,
        "down_spike_prices": down_prices,
        "entropy_dates": entropy_dates,
        "entropy_values": entropy_values,
        "summary": {
            "ticker": ticker,
            "total_days": total_days,
            "up_spikes": len(up_indices),
            "down_spikes": len(down_indices),
            "up_rate": up_rate,
            "down_rate": down_rate
        }
    }
