"""
ST-segment anaysis functions
"""

import math
import numpy as np
import neurokit2 as nk
import logging

logger = logging.getLogger(__name__)


def find_r_peaks(ecg, leads=None, sampling_rate=1000):
    """
    Find ECG peaks in the provided ECG data.

    This function identifies ECG peaks by analyzing the 'mv' (millivolts) column of the ECG DataFrame.

    Parameters
    ----------
    ecg : pandas.DataFrame
        A DataFrame containing ECG data with 'mv' (millivolts) column.
    leads: List[str]
        An optional list of lead names to find R-peaks for.
        All leads will be used by default.
    sampling_rate : int
        The sampling rate of the ECG data. Default is 1000 Hz.

    Returns
    -------
    peaks : dict
        A dictionary containing R-peaks time in milliseconds for each lead.
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

    peaks = {}

    for lead in leads:
        _, r_peaks = nk.ecg_peaks(
            ecg[lead],
            sampling_rate=sampling_rate,
            method="martinez2004",
            correct_artifacts=False,
            show=False,
        )
        peaks[lead] = r_peaks["ECG_R_Peaks"] / r_peaks["sampling_rate"] * 1000

    return peaks


def calculate_averaged_rr_interval_for_leads(r_peaks_times_all_leads):
    """
    Calculate the average RR interval for each lead from R-peaks time data.

    This function calculates the average RR interval for each lead by computing the time differences
    between consecutive R-peaks, filtering out intervals that are not within the range
    of 200 to 2000 milliseconds. This wide range reflects a more inclusive population, such as outliers and
    well-trained athletes with lower heart rates. As a reference, a normal resting heart rate is 
    60 bpm (1000 milliseconds) to 100 bpm (600 milliseconds).

    Parameters
    ----------
    R_peaks_times_all_leads : dict
        A dictionary containing R-peaks times in milliseconds for all leads, where each key represents a lead
        and the corresponding value is a list or array-like containing the times of R-peaks for that lead.

    Returns
    -------
    average_rr_intervals : dict
        A dictionary containing the average RR interval in seconds for each lead.
    """
    average_rr_intervals = {}

    for lead, r_peaks_time in r_peaks_times_all_leads.items():
        rr_interval = []
        for i in range(len(r_peaks_time)-1):
            rr_interval.append(r_peaks_time[i+1] - r_peaks_time[i])

        rr_interval = [x for x in rr_interval if 200 <= x <= 2000]

        if len(rr_interval) == 0:
            average_rr_intervals[lead] = np.nan
        else:
            average_rr_intervals[lead] = np.mean(rr_interval) / 1000

    return average_rr_intervals


def calculate_heart_rate(rr_values_dict):
    """
    Calculate heart rate from RR interval values.

    This function calculates heart rate by taking the reciprocal of RR interval values provided.

    Parameters
    ----------
    rr_values_dict : dict
        A dictionary containing RR interval values in seconds for each lead.

    Returns
    -------
    heart_rates : dict
        A dictionary containing calculated heart rates in beats per minute (bpm) for each lead.
    """
    heart_rates = {}
    for lead, rr_values in rr_values_dict.items():
        heart_rate = 60 / rr_values
        heart_rates[lead] = heart_rate
    return heart_rates


def calculate_time_ms_between_jpoint_and_twave(j_point_indices, t_wave_onset_indices, sampling_rate=1000):
    """
    Calculate the time between J-points and T-wave onsets.

    Parameters
    ----------
    j_point_indices : list
        A list indexing the time where J-points are detected in the ECG signal.
    t_wave_onset_indices : list
        A list indexing the time where T-wave onsets are detected in the ECG signal.
    sampling_rate : int
        The sampling rate of the ECG data. Default is 1000 Hz.

    Returns
    -------
    time_differences_ms : list
        A list containing the time differences in milliseconds (ms) between a J-points and T-wave onsets.
    """

    # Check if both J-point and T-wave onset indices are provided and have the same length
    if len(j_point_indices) != len(t_wave_onset_indices):
        raise ValueError("The number of J-points and T-wave onsets must be the same.")


    time_differences_ms = []

    for j_point, t_onset in zip(j_point_indices, t_wave_onset_indices):
        # Calculate the time difference in samples
        time_difference_samples = t_onset - j_point

        # Convert to milliseconds
        time_difference_ms = (time_difference_samples / sampling_rate) * 1000

        time_differences_ms.append(time_difference_ms)

    return time_differences_ms


def calculate_stc(st_ms, rr_interval):
    """
    Calculate the STc value for each ST segment duration based on the RR interval.

   Parameters:
    st_ms (list): List of ST segment durations in milliseconds (ms).
    rr_interval (float): RR interval in seconds.

   Returns:
    list: List of STC values (ms).
    """

    # Ensure that rr_interval is a non-zero positive value to avoid division errors
    if rr_interval <= 0:
        raise ValueError("RR interval must be a positive number")

    # Calculate STc for each ST_ms value
    stc_values = [st / math.sqrt(rr_interval) for st in st_ms]

    return stc_values


def detect_r_peak_offset(ecg_net_positive, r_peak_indices, sampling_rate=1000):
    """
    Delineate the ECG signals and extract R-offsets (i.e. end of the QRS complexes).

    This function uses discrete wavelet transform (DWT) to delineate the ECG signals
    for the given leads and extract the R-offsets from the delineated waves.

    Parameters
    ----------
    ecg_net_positive : pandas.DataFrame
        A DataFrame containing the predominantly upright (net positive deflection) ECG data in millivolts for the leads.
    r_peak_indices : list
        A list indexing the time where R-peaks are detected in the ECG signal,
        this is in milliseconds if the sampling rate is 1000 Hz.
    sampling_rate : int
        The sampling rate of the ECG data. Default is 1000 Hz.

    Returns
    -------
    r_offsets_indices : list
        A list indexing the time of the R-offsets for the given ECG lead.
    """
    try:
        _, waves_dwt = nk.ecg_delineate(
            ecg_net_positive,
            r_peak_indices,
            sampling_rate=sampling_rate,
            method="dwt",
            show=False,
            show_type="bounds_R",
        )

        if waves_dwt is not None:
            r_offsets_indices = waves_dwt.get("ECG_R_Offsets", [])
        else:
            r_offsets_indices = []
    except Exception as e:
        logger.error(f"Error delineating ECG: {e}")
        r_offsets_indices = []

    return r_offsets_indices


def get_threshold_for_lead_age_sex(lead, age, sex):
    """
    Determine the horizontal line threshold based on lead, age, and sex.

    Args:
    - lead (str): The ECG lead identifier.
    - age (int): Age of the subject.
    - sex (str): Sex of the subject ('male' or 'female').

    Returns:
    - float: Threshold value in millivolts (mV) for the horizontal line.
    """
    threshold_value = 0.1
    if lead in ("v2", "v3"):
        if sex == "male":
            threshold_value = 0.25 if age < 40 else 0.2
        elif sex == "female":
            threshold_value = 0.15

    return threshold_value


def calculate_custom_baseline_value(signal):
    """
    Calculates a custom baseline value for a given signal by detrending and smoothing the signal,
    then computing its mean.

    Parameters:
    signal (array-like):
        The input signal data (e.g., a time series) for which the baseline is to be calculated.

    Returns:
    baseline (float):
        The calculated baseline value, representing the mean of the detrended signal.
    """

    # Detrend and smooth the signal to approximate baseline
    detrended_signal = np.asarray(nk.signal_detrend(signal))

    # Calculate baseline as the mean of the Detrend signal
    # Safely compute mean
    if detrended_signal.size == 0 or np.all(np.isnan(detrended_signal)):
        baseline = np.nan
    else:
        baseline = np.mean(detrended_signal)

    return baseline
