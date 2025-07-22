import numpy as np

class MarketEnv:
    def __init__(self, prices, spike_indices, entropy_values=None):
        self.prices = prices
        self.spike_indices = set(spike_indices)
        self.entropy_values = entropy_values if entropy_values is not None else [0] * len(prices)

        self.reset()

    def reset(self):
        self.trade_log = []
        self.day = 0
        self.position = None  # None, "long"
        self.buy_price = 0
        self.profit = 0
        self.history = []  # stores (state, action, reward)
        return self._get_state()

    def _get_state(self):
        is_spike = int(self.day in self.spike_indices) #checks if the current day is a spike day
        entropy = float(self.entropy_values[self.day]) if self.day < len(self.entropy_values) else 0 #gets the entropy for the current day
        return np.array([is_spike, entropy], dtype=np.float32)

    def step(self, action):
        """
        Actions:
        0 = hold
        1 = buy
        2 = sell
        """
        reward = 0
        done = False

        price = self.prices[self.day]
        reward = 0

        # --- BUY ---
        if action == 1:
            if self.position is None:
                self.buy_price = price
                self.position = "long"
                self.trade_log.append({"day": self.day, "price": price, "type": "buy"})
            # else: already holding, do nothing

        # --- SELL ---
        elif action == 2:
            if self.position == "long":
                reward = price - self.buy_price
                self.profit += reward
                self.profit = float(self.profit)
                self.position = None
                self.buy_price = 0
                self.trade_log.append({"day": self.day, "price": price, "type": "sell"})
            # else: not holding, do nothing

        # HOLD or invalid actions naturally result in reward = 0

       

        self.day += 1
        if self.day >= len(self.prices) - 1:
            done = True

        next_state = self._get_state()
        self.history.append((next_state, action, reward))

        return next_state, reward, done
