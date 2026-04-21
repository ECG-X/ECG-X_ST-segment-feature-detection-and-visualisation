"""
Common plotting functions
"""

import matplotlib.pyplot as plt


def plot_ecg_lead_i_and_ii(signal_data, title):
    """
    Plot the first two ECG signal leads (i and ii).

    Args:
    - signal_data (dict): Dictionary containing ECG signal data.
        Keys are lead names (str) and values are signal arrays (array-like).
    - title (str): Title of the plot.

    Returns:
    - None
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot lead i
    axs[0].plot(signal_data["i"], label="Lead i")
    axs[0].set_ylabel("ECG Amplitude")
    axs[0].set_title("Lead i")

    # Plot lead ii
    axs[1].plot(signal_data["ii"], label="Lead ii")
    axs[1].set_ylabel("ECG Amplitude")
    axs[1].set_title("Lead ii")

    axs[1].set_xlabel("Sample")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
