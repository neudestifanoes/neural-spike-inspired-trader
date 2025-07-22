import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def calculate_entropy(spike_window):
    """
    Calculates the Shannon entropy of a window of binary spikes.
    """
    total = len(spike_window)
    if total == 0:
        return 0

    p1 = sum(spike_window) / total  # probability of spike (1) or it counts the 1s and divides by the total elements in the window
    p0 = 1 - p1                     # probability of no spike (0)

    entropy = 0
    if p1 > 0:
        entropy -= p1 * math.log2(p1)    #Shannon's enropy to see if the spikes are predictable in any way, great for binary data, low entropy means very predictable and high entropy is unpredictable
    if p0 > 0:
        entropy -= p0 * math.log2(p0)

    return entropy
