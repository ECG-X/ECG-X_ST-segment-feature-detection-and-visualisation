"""
Common signal processing functions
"""

from scipy import signal
import neurokit2 as nk
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def correct_baseline_wandering(
    ecg_mv,
    ecg_time,
    order=4,
    cutoff_frequency=0.5,
    sampling_rate=1000,
    plot_result=True,
):
    """
    Correct baseline wandering in ECG data for all 12 leads using a high-pass Butterworth filter.

    Parameters
    ----------
    ecg_mv : pandas.DataFrame
        DataFrame containing ECG signal in millivolts (mV) data for all 12 leads.
    ecg_time : numpy.ndarray
        The corresponding time data in seconds for the ECG signal.
    order : int, optional
        The filter order for the Butterworth filter. Default is 4.
    cutoff_frequency : float, optional
        The cutoff frequency for the high-pass filter in Hz. Default is 0.5 Hz.
    sampling_rate : int, optional
        The sampling rate of the ECG signal. Default is 1000 Hz.
    plot_result : bool, optional
        If True, plot the original and filtered signals for each lead. Default is True.

    Returns
    -------
    ecg_mv : pandas.DataFrame
        DataFrame containing the corrected ECG signal data (mV) for all 12 leads.

    """
    b, a = signal.butter(order, cutoff_frequency, btype="highpass", fs=sampling_rate)

    # Apply the high-pass filter to each lead column
    for column in ecg_mv.columns:
        if len(ecg_mv[column]) == 0 or ecg_mv[column].isna().all():
            continue  # skip empty or all-NaN columns
        original_signal = ecg_mv[column].copy()  # Store the original signal
        ecg_mv.loc[:, column] = signal.filtfilt(b, a, ecg_mv[column])
        if plot_result:
            # Plot the original and filtered signals
            _, axs = plt.subplots(2, 1, figsize=(15, 10))
            axs[0].plot(ecg_time, original_signal)
            axs[1].plot(ecg_time, ecg_mv[column])
            axs[0].set_xlabel("Time (s)")
            axs[1].set_xlabel("Time (s)")
            axs[0].set_title("Original Wandering Signal " + column)
            axs[1].set_title("Butterworth Filtered Signal " + column)
            axs[0].set_ylim([-3, +3])
            axs[1].set_ylim([-3, +3])
            plt.tight_layout()
            plt.show()

    return ecg_mv


def remove_noise(
    ecg_mv,
    ecg_time,
    sampling_rate=1000,
    lowcut=0.5,
    highcut=50.0,
    order=1,
    plot_result=True,
):
    """
    Apply a Butterworth filter to an ECG signal.

    Args:
    - ecg_mv (pandas.DataFrame): DataFrame containing ECG signal data in millivolts (mV) for all 12 leads.
    - ecg_time (numpy.ndarray): The corresponding time data in seconds for the ECG signal.
    - sampling_rate (float): Sampling rate of the ECG signal. Default is 1000 Hz.
    - lowcut (float, optional): Lower cutoff frequency in Hz. Default is 0.5 Hz.
    - highcut (float, optional): Upper cutoff frequency in Hz. Default is 50.0 Hz.
    - order (int, optional): Filter order. Default is 1.
    - plot_result (bool, optional): If True, plot the original and filtered signals for each lead.
        Default is True.

    Returns:
    - ecg_mv (pandas.DataFrame): DataFrame containing the filtered ECG signal data (mV) for all 12 leads.
    """
    # Normalize the cutoff frequencies by the Nyquist frequency
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq

    # Design the Butterworth filter
    b, a = signal.butter(order, [low, high], btype="band")

    # Apply the filter to each lead column
    for column in ecg_mv.columns:
        if len(ecg_mv[column]) == 0 or ecg_mv[column].isna().all():
            continue  # skip empty or all-NaN columns
        original_signal = ecg_mv[column].copy()  # Store the original signal
        ecg_mv.loc[:, column] = signal.filtfilt(b, a, ecg_mv[column])
        if plot_result:
            # Plot the original and filtered signals
            _, axs = plt.subplots(2, 1, figsize=(15, 10))
            axs[0].plot(ecg_time, original_signal)
            axs[1].plot(ecg_time, ecg_mv[column])
            axs[0].set_xlabel("Time (s)")
            axs[1].set_xlabel("Time (s)")
            axs[0].set_title("Original Noised Signal " + column)
            axs[1].set_title("Denoised Signal " + column)
            axs[0].set_ylim([-3, +3])
            axs[1].set_ylim([-3, +3])
            plt.tight_layout()
            plt.show()

    return ecg_mv


def read_and_invert_12_lead_ecg_data(
    ecg,
    ecg_time,
    sampling_rate=1000,
    leads=None,
):
    """
    Read ECG data, invert the signal if the raw signal is inverted, otherwise the signal is kept unchanged,
    and return as a DataFrame.

    Parameters
    ----------
    ecg : pandas.DataFrame
        A DataFrame containing ECG data in millivolts (mV).
    ecg_time : pandas.DataFrame
        A DataFrame containing ECG time data in seconds (s).
    sampling_rate : int
        The sampling rate of the ECG data. Default is 1000 Hz.
    leads : list of str (optional)
        The list of lead names to extract and check for inversion.
        All leads will be used by default.

    Returns
    -------
    ecg_net_positive : pandas.DataFrame
        A DataFrame containing the combined ECG data (mV) with columns "time" (s)
        and the specified 12 leads with predominantly upright signals (net postive deflection).
    """

    # Set default leads names if none were provided
    if leads is None:
        leads = [
            "i",
            "ii",
            "iii",
            "avr",
            "avl",
            "avf",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
        ]

    # Check if all specified leads are in the DataFrame
    missing_leads = [lead for lead in leads if lead not in ecg.columns]
    if missing_leads:
        raise KeyError(
            f"The following leads are missing from the ECG data: {missing_leads}"
        )

    # Extract the specified leads
    leads_data = ecg[leads].copy()  # Use copy to avoid modifying the original DataFrame

    # Check for time column in ecg_time
    if "s" not in ecg_time.columns:
        raise KeyError("The time column 's' is missing from ecg_time DataFrame.")

    # Assign ECG time data
    leads_data["time"] = ecg_time["s"].values  # Ensure proper alignment

    # Invert the signals (if the raw signal is inverted) directly within the function
    ecg_net_positive = leads_data.copy()
    for lead in leads:
        if lead in ecg_net_positive.columns:
            try:
                ecg_net_positive[lead], _ = nk.ecg_invert(
                    ecg_net_positive[lead], sampling_rate=sampling_rate, show=False
                )
            except Exception as e:
                logger.error(f"Error inverting lead '{lead}': {e}")
                # Optionally handle the error, e.g., continue or raise an exception
        else:
            logger.info(f"Lead '{lead}' not found in the DataFrame after extraction.")

    return ecg_net_positive
