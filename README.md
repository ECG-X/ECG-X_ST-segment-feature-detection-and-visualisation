# ECG-X ST Segment Feature Detection and Visualisation

This repository contains code used to generate stimuli used in the experimental work described in _Increasing the accuracy of ST-segment elevation detection in an ECG through automated visual rules_ [1].

The code implements a feature detection and visualisation pipeline, that uses data collected from [PhysioNet](https://physionet.org/) to identify ST-segment features. The pipeline takes in ECG signals, cleans them, extracts important features (e.g. **R-peaks** , **T-waves**, and **J-points**), and generates plots that highlight features relevant to ST-elevation interpretation.

## Quickstart

### 1. Setup

This project should be installed with uv (https://docs.astral.sh/uv). If you don't already have uv installed, please follow the [official documentation to install uv for your operating system](https://docs.astral.sh/uv/getting-started/installation/).

First clone the repository:

```bash
git clone https://github.com/ECG-X/ECG-X_ST-segment-feature-detection-and-visualisation.git
```

Then type the following to create a virtual environment and install all dependencies:

```bash
cd ECG-X_ST-segment-feature-detection-and-visualisation
uv sync
```

You can use `uv run` to invoke the scripts:

```bash
uv run python name-of-script.py
```

or, if you prefer, you can activate the virtual environment in the usual way:

```bash
source .venv/bin/activate
```

### 2. Input data

You'll need to download the PTB Diagnostic ECG Database and run a script to convert it into a form suitable for use with the feature detection and visualisation pipeline.

You can find the script in the [`data_prep`](data_prep/) directory. Full instructions are provided in the data_prep [`README`](data_prep/README.md).

### 3. Configuration

The ST-segment feature detection and visualisation pipeline is configuration driven, and there are configuration files in `pipeline/configs/`. For example [`src/configs/st_visualisation_myocardial.yaml`](src/configs/st_visualisation_myocardial.yaml) can be used to generate the ECG images for subjects with a myocardial infarction diagnosis.

See the [Pipeline configuration](#pipeline-configuration) section for a full explanation  of the various configuration settings.

### 4. Running the ST-segment feature detection and visualisation pipeline

To run the pipline use `uv run` to invoke the script, passing it a configuration file:

```bash
cd pipeline

# Run the pipeline on the myocardial dataset
uv run python st_visualisation.py --config-path=configs/st_visualisation_myocardial.yaml

# Run the pipeline on the control dataset
uv run python st_visualisation.py --config-path=configs/st_visualisation_control.yaml
```

This will generate plots of each time segment window on red ECG grid paper style background, with markers for detected features (J-points, thresholds, baselines, and shading (AUC) for elevated ST-segments).

By default these plots are saved in the `plots/` directory. 

See [The ECG-X ST-segment feature detection and visualisation pipeline](#the-ecg-x-st-segment-feature-detection-and-visualisation-pipeline) section for a full explanation of the steps performed by the pipeline.

## Pipeline configuration

There are configuration files for controlling the The ST-segment feature detection and visualisation pipeline in pipeline/configs. These configuration files were used to generate the plots used in the experiment described in  _Increasing the accuracy of ST-segment elevation detection in an ECG through automated visual rules_ [1].

This section gives details of the various configuration settings.

### data section:

* `input_dir` is the path to the input data files.
* `output_dir` is the path to the location where the pipeline output plots should be saved.
* `leads` specifies which leads to consider when generating the ECG images.

### plot section:
* `sampling_rate` this must match the sampling rate given in your dataset description e.g., in Physionet for the [PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/) the provided sampling rate is 1000 (samples per second). If other sampling rate comes with your dataset description, set it accordingly.

* `segment_duration_sec` determines the time segment window of a ECG plot in seconds.

  * The default value is 5s.

* `total_duration_sec` determines the total duration of the ECG signal sampled as input and plotted as output in seconds.

  * The default value is 30s.

  * **Note** The default values of `segment_duration_sec` (5s) & `total_duration_sec` (30s) will result in a 30 second ECG split into six 5-second plots.

* `mm_per_second` sets the horizontal scale of the ECG plot in millimeters per second.

  * The default value is 25 mm/s.

* `mm_per_mv` sets the vertical scale in millimeters per millivolt.

  * The default value is set to 10 mm/mV.

* `max_amplitude` determines the maximum acceptable ECG signal amplitude in millivolts.

  * Any signal exceeding ±3 mV will be skipped to avoid plotting artifacts or noisy signals.

* `show_custom_baseline` toggles whether the calculated baseline (isoelectric line) is drawn on the ECG plot.

  * Helpful for checking baseline correction accuracy.

* `show_black_threshold_line` toggles whether the threshold line (baseline + threshold) is shown.

  * Useful for visualizing the decision boundary used in ST-elevation detection.

* `fill_auc` toggles whether to shade the area under the curve (AUC) for ST-segment regions.

  * Provides visual emphasis of elevated segments.

* `show_axis_labels` determines if time (x-axis) and voltage (y-axis) labels are displayed.

  * Useful for context but can be hidden for cleaner images.
  * **Note** the labels are only shown if `occupy_full_fig_canvas` is set to `false`.

* `show_inline_plot` toggles inline visualization.

  * When `true`, plots are displayed interactively (e.g., in a notebook).
  * When `false`, plots are only saved to disk.

* `occupy_full_fig_canvas` determines whether the ECG plot uses the full figure canvas.

  * When `true`, reduces white space and maximizes the grid display.

* `save_plot` toggles automatic saving of plots to the output directory.

  * If disabled, plots are shown inline (if enabled) but not saved.

### debug section:

* `plot_intermediate_ecgs` toggles plotting at intermediate preprocessing stages (e.g., after baseline correction or noise removal).

  * When `true`, helps verify signal cleaning steps visually.
  * When `false`, only final output plots are generated.


## The ECG-X ST-segment feature detection and visualisation pipeline

Below is an explanation of the steps performed in the pipeline:

### 1. ECG Time Column Generation
After organizing your data, a **time column** is generated for the ECG signals.
This aligns data points with their corresponding time stamps based on:
- **Sampling rate** in samples per second or Hertz (Hz)
- **Total number of samples**

**Formula:**
```
Time Interval = 1 / Sampling Rate
```

For example, with a sampling rate of 500 Hz over 10 seconds, you get 5000 samples. Each sample represents an interval of 1/500 seconds, which builds the time column in seconds.
This time column is essential for visualizing ECG signals and detecting anomalies accurately.


### 2. Reading the Signal
- ECG signals are loaded from `.csv` files in the input directory.
- Required leads are selected based on configuration.
- The ECG time column is added.
- Optional patient metadata (age, sex) is read when needed.


### 3. Correcting Baseline Wander
- Slow baseline drift caused by respiration or movement is removed.
- Filtering techniques align the baseline closer to zero, improving feature detection accuracy.


### 4. Removing Noise
- Noise sources such as **powerline interference** or **muscle activity** are filtered out.
- Produces a cleaner signal for feature extraction.


### 5. Inverting the Signal When Necessary
- Each lead is checked for potential inversion (in step 6 and step 9).
- Inverted raw (i.e., recorded) ECG signals, sometimes due to artefacts such as electrode placement or pathological causes, are identified and inverted. Otherwise, the signal is kept unchanged.
- This ensures that the net signal, of each of the 12 leads, are predominantly upright (positive deflection), improving R-peak detection reliability.


### 6. Finding R-peaks (QRS Complex Detection)
- R-peaks are detected from the lead III ECG signal (see step 5).
- Used to compute:
  - Average RR interval (s)
  - Heart rate (bpm)


### 7. Signal Quality Check
- Amplitude of each lead is validated against configured maximum limits.
- Files with excessively large or noisy signals are skipped.


### 8. Defining Custom Baseline and Thresholds
- A baseline is calculated for each lead.
- Thresholds are adjusted based on patient metadata (age, sex) when applicable.
- A **black threshold** is computed as:
`black_threshold = baseline + threshold`


### 9. Segment-wise Processing
Each ECG lead is divided into **fixed-length segments** (duration defined in config).

For each segment (see step 5):
- **J-point detection**: Identifies the offset of the R-peak, defined by the end of the QRS complex.
- **T-wave delineation**: Detects the onset of the T-wave using wavelet-based methods.
- **ST interval**: Time between J-point and T-wave onset (ms).
- **STc calculation**: Corrected ST interval based on the average RR interval.
- **Segment validation**: Skips invalid segments (NaN, flatline, too short).


### 10. Plotting ECG with Identified Features
- Segments are plotted on **red mm-grid paper style background**.
- Plots show:
  - Filtered ECG signals
  - J-points
  - Shading: filled area under the curve to highlight elevated ST-segment
  - Baseline and threshold lines
  - R-peaks (not shown by default)
  - Inline or interactive plots can be enabled for debugging, which shows additional features such as T-wave onsets and offsets.
## ⚖️ License

Software in this repository is made available under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) licence.

>[!NOTE]
> Commercial use is not permitted under this licence. Any use of this software or derivative works for commercial purposes - including use by for‑profit organisations, integration into commercial products or services, or monetisation in any form - requires a separate licence agreement.
> For commercial licensing enquiries, please contact The Innovation Factory, The University of Manchester: [licensing@uominnovationfactory.com](licensing@uominnovationfactory.com) 

For more information, see [LICENSE.md](LICENSE.md).

## Authors

This codebase was primarily implemented by [Adina Rahim](https://github.com/Adina-Rahim), who developed the core data processing and ST-segment feature detection and visualisation pipeline, including the end-to-end signal processing workflow.

See [AUTHORS.md](AUTHORS.md) for a full list of contributors to this project and their [CRediT](https://credit.niso.org) contributions.

## References

[1] L. Hughes-Noehrer, G. Strain, A. Rahim, J. Sinnott,
J. Carlton, C. T. Wu, R. Body, C. Jay, Increasing the accuracy of ST-segment elevation detection in an ECG through automated visual rules [Submitted] (2026), [Preprint (Version 1)](https://doi.org/10.21203/rs.3.rs-9537195/v1)