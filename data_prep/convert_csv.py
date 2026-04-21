"""
This script converts ECG data from .dat files into .csv format, adding a time column based on the provided sampling rate. 
It reads ECG signals using the `wfdb` library, processes them into a DataFrame, and writes the resulting data to a CSV file.
"""

import logging
import wfdb
import numpy as np
import pandas as pd
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_dat_to_csv(dat_file_path: str, output_csv_path: str, sampling_rate: int) -> None:
    """
    Converts a .dat ECG file to a .csv file with time and signal data.
    
    Args:
        dat_file_path (str): Path to the input `.dat` file. This should point to a valid `.dat` file.
        output_csv_path (str): Path to the output `.csv` file where the ECG data will be saved.
        sampling_rate (int): The sampling rate (in Hz) of the ECG data. It is used to generate the time axis
                              for the ECG signals. This value is typically available in the dataset's metadata.

    Returns:
        None: The function does not return any value. It writes the converted data to a CSV file.

    """
    # Read the data from the .dat file using wfdb (potential error source)
    try:
        signals, fields = wfdb.rdsamp(dat_file_path[:-4], sampfrom=0, sampto=None)
        sig_name = fields['sig_name']
    except FileNotFoundError:
        logger.error(f"File {dat_file_path} not found.")
        return
    except Exception as e:
        logger.error(f"Error reading {dat_file_path}: {e}")
        return
        
    # Convert to a DataFrame with each lead in a separate column
    data = pd.DataFrame(data=signals, columns=[name.lower() for name in sig_name])

    # Create a time axis based on the number of samples
    time = np.arange(len(data)) / sampling_rate  # assuming sampling rate is provided

    # Add time as the first column
    data.insert(0, 'Time (s)', time)

    # Ensure the directory for the CSV file exists
    output_directory = os.path.dirname(output_csv_path)
    os.makedirs(output_directory, exist_ok=True)

        # Save the DataFrame as a CSV (potential error source)
    try:
        data.to_csv(output_csv_path, index=False)
        logger.info(f"Converted {dat_file_path} to {output_csv_path}")
    except Exception as e:
        logger.error(f"Error writing {output_csv_path}: {e}")
