"""
Common data processing functions
"""

import pandas as pd


def generate_ecg_time_column(sampling_rate=1000, num_samples=38401):
    """
    Generate a time column as a DataFrame for ECG data.

    Parameters:
    sampling_rate (int): Sampling rate in samples per second (default is 1000 Hz).
    num_samples (int): Number of samples to generate (default is 38401).

    Returns:
    pd.DataFrame: DataFrame containing the time column with 's' (seconds) as the column name.
    """
    # Calculate the time interval
    time_interval = 1 / sampling_rate

    # Generate the time column
    time_column = [i * time_interval for i in range(num_samples)]

    # Convert the time column to a DataFrame
    ecg_time = pd.DataFrame({"s": time_column})

    return ecg_time



def read_12_lead_ecg_data(
    file_path,
    ecg_time,
    leads=None,
    read_metadata=True
):
    """
    Read ECG data from a CSV file and return it as a DataFrame, along with age and sex information.

    This function reads a CSV file containing ECG data, associates it with the provided ECG time
    data, and returns the combined data as a DataFrame with columns "time" and the specified 12
    leads.  Additionally, if read_metadata is True, it extracts 'age' and 'sex' information from
    the second row (index 1) of the CSV.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing ECG data.
    ecg_time : pandas.DataFrame
        A DataFrame containing ECG time data.
    leads: List[str]
        An optional list of lead names to extract.
        All leads will be used by default.
    read_metadata: bool, optional
        Specify whether to read 'age' and 'sex' from the csv

    Returns
    -------
    leads_data : pandas.DataFrame
        A DataFrame containing the combined ECG data with columns "time" and the specified 12 leads.
    age : int or None
        Age of the subject, extracted from the CSV (if available).
    sex : str or None
        Sex of the subject, extracted from the CSV (if available).
    """

    # Set default lead names if none were provided
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

    # Read ECG data from CSV
    ecg_data = pd.read_csv(file_path)

    # Read age/sex only if read_metadata is True
    age = ecg_data.loc[1, 'Age'] if read_metadata and 'Age' in ecg_data.columns and 1 in ecg_data.index else None
    sex = ecg_data.loc[1, 'Sex'] if read_metadata and 'Sex' in ecg_data.columns and 1 in ecg_data.index else None

    leads_data = ecg_data.loc[:, leads].copy()
    leads_data['time'] = ecg_time["s"]

    return leads_data, age, sex
