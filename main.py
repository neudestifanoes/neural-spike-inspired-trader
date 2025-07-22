# === IMPORTS ===
from Entropy.entropy_core import calculate_entropy  # Used by analysis internally
from utils.analysis import analyze_single_stock
from utils.simulator import simulate_spike_trading
from utils.market_env import MarketEnv
from agents.dqn_agent import DQNAgent
from pathlib import Path

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
tickers = ["AAPL", "TSLA", "GOOGL", "SPY"]
threshold = 0.01
window_size = 20
start_date = "2023-01-01"
end_date = "2023-12-31"

# === AAPL ANALYSIS ===
aapl = analyze_single_stock("AAPL", threshold, window_size, start_date, end_date)
aapl_spike_indices = sorted(
    [aapl["dates"].get_loc(d) for d in aapl["up_spike_dates"] + aapl["down_spike_dates"]]
)

# === SPIKE + ENTROPY VISUALIZATION ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Price & Spikes
ax1.plot(aapl["data"].index, aapl["data"]["Close"], label="AAPL Price", color="black")
ax1.scatter(aapl["up_spike_dates"], aapl["up_spike_prices"], color="green", label="Gains (≥1%)")
ax1.scatter(aapl["down_spike_dates"], aapl["down_spike_prices"], color="red", label="Losses (≤-1%)")
ax1.set_title("AAPL Price with Spike Overlays")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True)

summary = aapl["summary"]
info_text = (
    f"Total trading days: {summary['total_days']}\n"
    f"Upward spikes (≥1%): {summary['up_spikes']} ({summary['up_rate']:.2f}%)\n"
    f"Downward spikes (≤−1%): {summary['down_spikes']} ({summary['down_rate']:.2f}%)"
)
ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Bottom: Entropy
ax2.plot(aapl["entropy_dates"], aapl["entropy_values"], color='purple', label='Entropy (AAPL)')
ax2.set_title("Spike Train Entropy Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Entropy (bits)")
ax2.set_ylim(0, 1)
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()

# === ENTROPY COMPARISON ACROSS TICKERS ===
plt.figure(figsize=(14, 5))
ticker_data = {}

for ticker in tickers:
    ticker_data[ticker] = aapl if ticker == "AAPL" else analyze_single_stock(ticker, threshold, window_size, start_date, end_date)
    plt.plot(ticker_data[ticker]["entropy_dates"], ticker_data[ticker]["entropy_values"], label=f"{ticker}")

plt.title(f"Spike Entropy Comparison (Window={window_size}, Threshold={threshold*100:.0f}%)")
plt.xlabel("Date")
plt.ylabel("Entropy (bits)")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === SPIKE RATE BAR CHART ===
up_rates = [ticker_data[t]["summary"]["up_rate"] for t in tickers]
down_rates = [ticker_data[t]["summary"]["down_rate"] for t in tickers]
x = np.arange(len(tickers))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, up_rates, width, label='Up Spikes (%)', color='green')
plt.bar(x + width/2, down_rates, width, label='Down Spikes (%)', color='red')
plt.ylabel('Spike Rate (%)')
plt.title('Up vs Down Spike Rates by Stock')
plt.xticks(x, tickers)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === RULE-BASED ENVIRONMENT SIMULATION ===
print("\n[ENV] Dumb Agent Evaluation")
trades, profit_curve = simulate_spike_trading(aapl["prices"], aapl_spike_indices)
print(f"Simulated {len(trades)} trades. Total profit: ${float(profit_curve[-1]):.2f}")

env = MarketEnv(
    prices=aapl["prices"],
    spike_indices=aapl_spike_indices,
    entropy_values=aapl["entropy_values"]
)

state = env.reset()
done = False
while not done:
    is_spike = state[0]
    holding = env.position == "long"
    action = 1 if is_spike and not holding else 2 if is_spike and holding else 0
    state, reward, done = env.step(action)

print(f"[ENV] Final Profit from environment agent: ${float(env.profit):.2f}")

# === TRADE VISUALIZATION ===
plt.figure(figsize=(14, 5))
plt.plot(aapl["dates"], np.array(aapl["prices"]).flatten()[1:], label="Price", color="black")

buy_days = [t["day"] for t in env.trade_log if t["type"] == "buy"]
sell_days = [t["day"] for t in env.trade_log if t["type"] == "sell"]
buy_prices = [t["price"] for t in env.trade_log if t["type"] == "buy"]
sell_prices = [t["price"] for t in env.trade_log if t["type"] == "sell"]
buy_dates = [aapl["dates"][i] for i in buy_days]
sell_dates = [aapl["dates"][i] for i in sell_days]

plt.scatter(buy_dates, buy_prices, color="green", marker="^", label="Buy")
plt.scatter(sell_dates, sell_prices, color="red", marker="v", label="Sell")
plt.title("Agent Trades Over Price Chart")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === DQN TRAINING ===
print("\n=== Training Deep Q-Learning Agent ===")

env.reset()
agent = DQNAgent(state_dim=2, action_dim=3)
num_episodes = 50
profits = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    profits.append(float(env.profit))
    print(f"Episode {episode+1}/{num_episodes} → Profit: ${env.profit:.2f}, Epsilon: {agent.epsilon:.3f}")

# === Save model after training ===
agent.save_model("model/dqn_aapl_2023.pth")
print("[✓] Model saved to model/dqn_aapl_2023.pth")

# === DQN PERFORMANCE PLOT ===
plt.figure(figsize=(10, 4))
plt.plot(profits, label="Profit per Episode")
plt.title("Deep Q-Learning Agent Performance")
plt.xlabel("Episode")
plt.ylabel("Profit ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === CUMULATIVE TRAINING ACROSS STOCKS AND YEARS ===

print("\n=== Multi-Stock Multi-Year Cumulative Training ===")

# Custom runs: add more as needed
# most similar and meaningfully overlapped companies in sector plus Nvidia for volatility
training_runs = [("AAPL", 2021), ("MSFT", 2022), ("NVDA", 2023)]
cumulative_model_path = "model/cumulative_trader.pth"

# Load existing model if available
cumulative_agent = DQNAgent(state_dim=2, action_dim=3)
if Path(cumulative_model_path).exists():
    cumulative_agent.load_model(cumulative_model_path)
    print(f"[✓] Loaded model from {cumulative_model_path}")

# Train on each (ticker, year) pair
for stock, year in training_runs:
    print(f"\n→ Training on {stock} ({year})")
    data = analyze_single_stock(stock, threshold, window_size, f"{year}-01-01", f"{year}-12-31")
    spike_indices = [data["dates"].get_loc(d) for d in data["up_spike_dates"] + data["down_spike_dates"]]
    env = MarketEnv(data["prices"], spike_indices, data["entropy_values"])

    for ep in range(10):  # training episodes per stock
        state = env.reset()
        done = False
        while not done:
            action = cumulative_agent.select_action(state)
            next_state, reward, done = env.step(action)
            cumulative_agent.store(state, action, reward, next_state, done)
            cumulative_agent.train()
            state = next_state
        print(f"    Episode {ep+1}/10 → Profit: ${env.profit:.2f} | Epsilon: {cumulative_agent.epsilon:.3f}")

# Save updated model
cumulative_agent.save_model(cumulative_model_path)
print(f"[✓] Saved cumulative model to {cumulative_model_path}")


'''
# * EVALUATION ON UNSEEN DATA 

print("\n=== Evaluation on Unseen Data ===")

# Just change these 2 variables to try different stocks and years
eval_ticker = "AAPL"
eval_year = 2024

# Load the saved agent
eval_agent = DQNAgent(state_dim=2, action_dim=3)
eval_agent.load_model("model/cumulative_trader.pth")

# Load unseen data
eval_data = analyze_single_stock(
    eval_ticker, threshold, window_size,
    f"{eval_year}-01-01", f"{eval_year}-12-31"
)
spike_indices = [
    eval_data["dates"].get_loc(d)
    for d in eval_data["up_spike_dates"] + eval_data["down_spike_dates"]
]

# Run simulation using the loaded agent
env = MarketEnv(eval_data["prices"], spike_indices, eval_data["entropy_values"])
state = env.reset()
done = False

while not done:
    action = eval_agent.select_action(state)
    state, reward, done = env.step(action)

# Print results
print(f"[✓] Evaluation on {eval_ticker} ({eval_year}) → Final Profit: ${env.profit:.2f}")

# Plot trades
plt.figure(figsize=(14, 5))
plt.plot(eval_data["dates"], np.array(eval_data["prices"]).flatten()[1:], label="Price", color="black")

buy_days = [t["day"] for t in env.trade_log if t["type"] == "buy"]
sell_days = [t["day"] for t in env.trade_log if t["type"] == "sell"]
buy_prices = [t["price"] for t in env.trade_log if t["type"] == "buy"]
sell_prices = [t["price"] for t in env.trade_log if t["type"] == "sell"]
buy_dates = [eval_data["dates"][i] for i in buy_days]
sell_dates = [eval_data["dates"][i] for i in sell_days]

plt.scatter(buy_dates, buy_prices, color="green", marker="^", label="Buy")
plt.scatter(sell_dates, sell_prices, color="red", marker="v", label="Sell")
plt.title(f"{eval_ticker} ({eval_year}) Trades by Trained Agent")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''