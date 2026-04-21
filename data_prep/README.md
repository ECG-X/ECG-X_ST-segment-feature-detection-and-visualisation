# ECG Dataset Preparation

This directory contains a script which converts the ECG dataset from the PTB Diagnostic ECG Database into CSV files suitable for use with our visualisation pipeline.

## Prerequisite - Download the PTB Diagnostic ECG Database

Before you start you'll need to download and extract the PTB Diagnostic ECG Database from [PhysioNet](https://physionet.org/).

The URL of the database on PhysioNet is: https://physionet.org/content/ptbdb/1.0.0/

You can download the database as ZIP file using this URL: https://physionet.org/content/ptbdb/get-zip/1.0.0/

After downloading the PTB Diagnostic ECG Database ZIP file, extract it to the `data/` directory at the root level of this repo. Once extracted you should have a directory structure that looks something like this:

```
data/
└── ptb-diagnostic-ecg-database-1.0.0/
    ├── patient001
    ├── patient002
    ├── ...
    ├── patient294
    ├── CONTROLS
    ├── ...
```

## Setup

You should already have a virtual environment setup and ready to go; if not see the setup instructions in [the main readme](../README.md).

## Configuration 

The script is driven by a configuration YAML file (`config.yaml`). This file contains information such as the base directory of the PTB Diagnostic ECG Database and the output location for the converted CSV files.

The default configuration expects the base directory to match the directory structure described in [Prerequisite - Download the PTB Diagnostic ECG Database](#prerequisite-download-the-ptb-diagnostic-ecg-database).

## Running

To run the script, use the following command:

```bash
uv run python runner.py --config-path config.yaml
```

## Output

Running the script will generate CSV files and store them in `data/processed/`. Assuming you have used the default configuration, the records will be split into three groups:

- `myocardial` - patients with a myocardial infarction diagnosis.
- `control_group` - healthy patients.
- `other` - patients with other conditions (not used by the visualisation pipeline).

This will result in a directory structure that looks something like this:

```
data/
├── processed/
│    ├── control_group/
│    │   ├── patient104/
│    │   │   ├── s0306lre.csv
│    │   │   ├── ...
│    │   ├── ...
│    ├── myocardial/
│    │   ├── patient001/
│    │   │   ├── s0010_re.csv
│    │   │   ├── ...
│    │   ├── ...
│    └── other/
│        ├── patient106/
│        ├── ...
└── ptb-diagnostic-ecg-database-1.0.0/
```
