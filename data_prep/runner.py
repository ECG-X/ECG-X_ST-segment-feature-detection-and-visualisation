"""
This script processes ECG data by converting .dat files into .csv format using configuration from a YAML file.
If diagnostics_splitter is enabled, it first organizes patient folders by diagnosis
before converting .dat files into .csv in separate myocardial and control folders.
"""

import os
import logging
import argparse
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional

import convert_csv
import diagnosis_splitter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------
# Configuration Models
# ------------------------

class DataConfig(BaseModel):
    """
    Configuration model for specifying dataset paths.

    Attributes:
        base_directory (str): The root directory where the dataset files are located.
        record_file (str): The name of the specific record file to be processed.
    """
    base_directory: str
    record_file: str

class GeneralConfig(BaseModel):
    """
    General configuration settings for data processing.

    Attributes:
        sampling_rate (int): The sampling rate (in Hz) to use when processing ECG data.
    """
    sampling_rate: int
    separate_by_diagnosis: bool = False

class OutputConfig(BaseModel):
    """
    Configuration for specifying the output location.

    Attributes:
        output_directory (str): Directory where the converted CSV files will be saved.
    """
    output_directory: str

class Settings(BaseModel):
    """
    Top-level configuration model that aggregates all settings.

    Attributes:
        data (DataConfig): Configuration related to input dataset paths.
        general (GeneralConfig): General processing settings such as sampling rate.
        output (OutputConfig): Configuration for output file storage.
    """
    data: DataConfig
    general: GeneralConfig
    output: OutputConfig

# ------------------------
# Utility Functions
# ------------------------

def load_config(config_path: Path = Path("config.yaml")) -> Settings:
    """
    Load and parse the configuration file to create a Settings object.

    Args:
        config_path (Path): The path to the configuration file (default is "config.yaml").

    Returns:
        Settings: A Settings object populated with values from the configuration file.
    """
    with open(config_path, "r") as file:
        raw_config = yaml.safe_load(file)
    return Settings(**raw_config)

def read_record_file(record_file_path: str) -> List[str]:
    """
    Read records from a file and return them as a list of strings.

    Args:
        record_file_path (str): The path to the record file to be read.

    Returns:
        list: A list of records (each as a string), or an empty list if there was an error.
    """
    try:
        with open(record_file_path, "r") as file:
            records = [line.strip() for line in file.readlines()]
        logger.info(f"Read {len(records)} records from {record_file_path}")
        return records
    except FileNotFoundError:
        logger.error(f"File {record_file_path} not found.")
        return []
    except Exception as e:
        logger.error(f"Error reading the file {record_file_path}: {e}")
        return []

def build_output_dirs(output_directory: str, separate_by_diagnosis: bool) -> Dict[str, str]:
    """
    Create output folders and return a label-to-directory mapping.

    Args:
        output_directory (str): The base directory where output CSV files will be saved.
        separate_by_diagnosis (bool): Whether to create separate folders based on diagnosis labels.
                                      defaults to a single "raw" folder otherwise.

    Returns:
        dict: A mapping from diagnosis labels to their corresponding output directories.
    """
    if separate_by_diagnosis:
        logger.info("Diagnosis splitting ENABLED. Processing records by diagnosis...")
        label_to_dir = {
            "myocardial": os.path.join(output_directory, "myocardial"),
            "control_group": os.path.join(output_directory, "control_group"),
            "other": os.path.join(output_directory, "other"),
        }
    else:
        logger.info("Diagnosis splitting DISABLED. Converting all records to 'raw' folder.")
        label_to_dir = {"raw": os.path.join(output_directory, "raw")}

    for output_dir in label_to_dir.values():
        os.makedirs(output_dir, exist_ok=True)

    return label_to_dir

def get_label_for_record(base_directory: str, record: str) -> str:
    """
    Resolve diagnosis label using .hea metadata, with safe fallback to 'other'.

    Args:
        base_directory (str): The base directory containing the .hea files.
        record (str): The name of the record for which to determine the diagnosis label.

    Returns:
        str: The diagnosis label for the record.
    """
    hea_path = os.path.join(base_directory, f"{record}.hea")
    
    if not os.path.isfile(hea_path):
        logger.warning(f".hea file missing: {hea_path}. Treating as 'other'.")
        return "other"

    return diagnosis_splitter.classify_record(hea_path, diagnosis_keyword="myocardial")

def convert_record_to_csv(dat_path: str, output_path: str, sampling_rate: int, record: str, label: Optional[str] = None) -> None:
    """
    Convert ECG .dat record to CSV

    Args:
        dat_path (str): Path to the input .dat file.
        output_path (str): Path where the output CSV file will be saved.
        sampling_rate (int): The sampling rate of the ECG data.
        record (str): The name of the record being processed (used for logging).
        label (Optional[str]): optional diagnosis label for the record (used for logging).
    """
    try:
        convert_csv.convert_dat_to_csv(dat_path, output_path, sampling_rate)
        if label is None:
            logger.info(f"Converted {record} to {output_path}")
        else:
            logger.info(f"Converted {record} as '{label}' to {output_path}")
    except Exception as e:
        logger.error(f"Error converting {record}: {e}")

# ------------------------
# Main Processing Function
# ------------------------

def process_ecg_records(config: Settings) -> None:
    """
    This function processes ECG data by:
    1. Loading configuration data (base directory, record file, sampling rate, and output directory).
    2. Reading the list of ECG records from the provided `record_file`.
    3. Generating CSV files for each record by converting `.dat` files to `.csv` files.
    4. Saving the CSV files in the specified `output_directory`.

    Args:
        config (Settings): A `Settings` object containing the configuration for processing,
                            including the base directory, record file name, sampling rate, and output directory.
    """
    # Load configuration values
    base_directory = config.data.base_directory
    record_file = config.data.record_file
    sampling_rate = config.general.sampling_rate
    output_directory = config.output.output_directory
    separate_by_diagnosis = config.general.separate_by_diagnosis

    # Build output directories and get label-to-directory mapping
    label_to_dir = build_output_dirs(output_directory, separate_by_diagnosis)

    record_file_path = os.path.join(base_directory, record_file)
    records = read_record_file(record_file_path)
    
    for record in records:
        # Check .dat path exists before proceeding
        dat_path = os.path.join(base_directory, f"{record}.dat")
        if not os.path.isfile(dat_path):
            logger.warning(f".dat file missing: {dat_path}")
            continue

        if separate_by_diagnosis:
            # Determine label and output path based on diagnosis
            label = get_label_for_record(base_directory, record)
            output_dir = label_to_dir.get(label, label_to_dir["other"])
            output_path = os.path.join(output_dir, f"{record}.csv")
            convert_record_to_csv(dat_path, output_path, sampling_rate, record, label=label)
        else:
            output_path = os.path.join(label_to_dir["raw"], f"{record}.csv")
            convert_record_to_csv(dat_path, output_path, sampling_rate, record)

# ------------------------
# Entry Point
# ------------------------

def main():
    """Main runner function for the script."""
    parser = argparse.ArgumentParser(description="Convert PhysioNet .dat files to CSV format with diagnosis-based splitting.")
    parser.add_argument(
        "--config-path",
        help="Path to the configuration YAML file.",
        default="config.yaml",
    )
    args = parser.parse_args()

    configuration_path = Path(args.config_path)
    if not configuration_path.exists():
        logger.error(f"Config file not found: {configuration_path}")
        return

    try:
        settings = load_config(config_path=configuration_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    process_ecg_records(settings)

if __name__ == "__main__":
    main()
