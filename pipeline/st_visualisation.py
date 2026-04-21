"""
ST-segment feature detection and visualisation pipeline
"""

import os
import argparse
import numpy as np
import neurokit2 as nk
import logging
from pathlib import Path

from shared import signal_processing, data_processing
from shared.plot import plot_ecg_lead_i_and_ii
from shared.utils import get_csv_paths
from st_segment import analysis as st_segment_analysis
from st_segment import plot as st_elevation_plot
from config import load_config, Settings


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(config: Settings):

    """
    - Read raw ECG data from csv files
    - Correct baseline wander in ECG signals using a high pass Butterworth filter
    - Remove noise from ECG signals using a bandpass Butterworth filter
    - Compute custom baseline by detrending filtered signal, and compute threshold line based on baseline
    - Check ECG signal quality against the configured maximum amplitude
    - Invert filtered ECG signals if raw signal is inverted so the net signal is predominantly upright 
        (positive deflection), otherwise the signal is kept unchanged
        - Compute average RR interval and heart rate based on lead III
        - Identify J-points, defined by the end of QRS complexes (start of ST-segments)
        - Identify T-wave onsets (end of ST-segments)
        - Compute STc (corrected end time of ST-segments)
    - Plot filtered ECG signals (with original deflections) on red grid background, including visualisations:
        threshold line, J-points, and shading. R-peak markers are hidden by default.
    """

    # Get a list of the paths of all csv files in the data input dir
    myfiles = get_csv_paths(config.data.input_dir)

    # Specify which leads to use
    required_leads = config.data.leads

    # Signals where any lead exceeds this amplitude will be ignored
    max_amplitude = config.plot.max_amplitude

    # Calculate the number of segments to divide the signal into
    num_segments = int(
        config.plot.total_duration_sec // config.plot.segment_duration_sec
    )

    # Sample length of each segment
    samples_per_segment = (
        config.plot.segment_duration_sec * config.plot.sampling_rate
    )

    # Start and end indices for total_duration_sec seconds of data
    start_idx, end_idx = (
        0,
        config.plot.total_duration_sec * config.plot.sampling_rate,
    )

    # Generate a time column for the entire signal, given the sampling rate and total duration
    ecg_time = data_processing.generate_ecg_time_column(
        sampling_rate=config.plot.sampling_rate, num_samples=(end_idx - start_idx)
    )

    # Determine if age/sex is required
    # it is only required when v2 and v3 is there Lukas suggested to make piepline generic as we are going to consider age and sex from now onwards
    # because we are not going to consider v2 and v3 which requires age and sex but instead of me removing it completely It is good to make condition
    # and keep code incase in future reserachers plans changes
    requires_age_sex = any(lead in ['v2', 'v3'] for lead in required_leads)

    logger.info(f"start_idx: {start_idx}, end_idx: {end_idx}")
    logger.info(f"requires_age_sex: {requires_age_sex}")

    # Iterate over all ECG files
    for f in range(len(myfiles)):

        csv_file_name = os.path.basename(myfiles[f])

        logger.info(f"----------------- {csv_file_name} -----------------")

        # Read ECG data
        try:
            ecg, age, sex = data_processing.read_12_lead_ecg_data(
                myfiles[f], ecg_time, required_leads, read_metadata=requires_age_sex
            )
        except Exception as e:
            logger.info(f"Error reading ECG data: {e}. Skipping...")
            continue  # Skip this file and go to the next one

        # Validate age and sex only if required
        if requires_age_sex:
            # Check if age is a valid integer; skip if not
            if isinstance(age, str):
                if not age.isdigit():
                    raise ValueError(f"File '{csv_file_name}': v2 or v3 requires a valid integer age, but got '{age}'.")
                age = int(age)  # Convert to integer after confirming it’s numeric
            elif not isinstance(age, (int, np.integer)):
                raise ValueError(f"File '{csv_file_name}': v2 or v3 requires integer age, but got type {type(age)}.")
            # Validate sex
            if sex not in ["male", "female"]:
                raise ValueError(f"File '{csv_file_name}': v2 or v3 requires sex to be 'male' or 'female', but got '{sex}'.")
        else:
            age = None
            sex = None

        ecg_raw = ecg[start_idx:end_idx]

        if config.debug.plot_intermediate_ecgs:
            plot_ecg_lead_i_and_ii(ecg_raw, "Raw ECG")

        # Correct baseline wandering
        ecg_mv = signal_processing.correct_baseline_wandering(
            ecg_raw,
            ecg_time,
            sampling_rate=config.plot.sampling_rate,
            plot_result=config.debug.plot_intermediate_ecgs,
        )

        # Remove Noise
        filtered_ecgs = signal_processing.remove_noise(
            ecg_mv,
            ecg_time,
            sampling_rate=config.plot.sampling_rate,
            plot_result=config.debug.plot_intermediate_ecgs,
        )

        # Check if all required leads exist in the filtered data
        missing_leads = [lead for lead in required_leads if lead not in filtered_ecgs.columns]
        if missing_leads:
            logger.info(f"Missing required leads: {missing_leads}. Skipping...")
            continue

        # ------------------------ Heart Rate Calculation ------------------------
        # Use lead III signal to determine heart rate
        # Invert ECG signal if the raw signal is inverted, so the net signal is predominantly upright (positive deflection)
        ecg_net_positive = signal_processing.read_and_invert_12_lead_ecg_data(
            filtered_ecgs, ecg_time, sampling_rate=config.plot.sampling_rate, leads=['iii'])

        # Detect R-peaks in lead III
        r_peaks_time = st_segment_analysis.find_r_peaks(ecg_net_positive, leads= ['iii'], sampling_rate=config.plot.sampling_rate)
        ecg_net_positive = ecg_net_positive['iii'].tolist()

        logger.info(f"R-peaks time (ms):\n{r_peaks_time}")

        # Compute average RR intervals and heart rate from detected peaks
        average_rr_intervals = st_segment_analysis.calculate_averaged_rr_interval_for_leads(r_peaks_time)
        logger.info(f"Heart rate (bpm): {st_segment_analysis.calculate_heart_rate(average_rr_intervals)}")

        # Retrieve RR interval in seconds
        rr_s = average_rr_intervals['iii']
        logger.info(f"Mean RR interval (s): {rr_s}")

        # ------------------------ Signal Quality Check ------------------------
        # Check max signal amplitude of each lead and skip if it exceeds max_amplitude
        skip = False
        for lead in required_leads:
            # Get the maximum and minimum values of the ECG signal for the current lead
            max_val = filtered_ecgs[lead].max()
            min_val = filtered_ecgs[lead].min()

            # Check if the signal amplitude is outside of ±max_amplitude in millivolts (mV)
            if max_val > max_amplitude or min_val < -(max_amplitude):
                logger.info(
                    f"File skipped. Lead {lead} exceeds ±{max_amplitude} mV (max: {max_val},\
                        min: {min_val})."
                )
                skip = True
                break

        if skip:
            continue  # don't plot any of the leads for this file

        # ------------------------ Segment-wise Processing ------------------------
        # Loop over each lead in the ECG
        for lead in required_leads:
            logger.info(f"Processing lead: {lead}")
            signal = filtered_ecgs[lead]

            # Check if the signal is long enough
            if (
                len(signal)
                < config.plot.total_duration_sec * config.plot.sampling_rate
            ):
                logger.info(
                    f"Signal too short for {config['plot']['total_duration_sec']} seconds in lead \
                        {lead}. Skipping this lead."
                )
                continue

            # Extract threshold value for each lead based on age/sex rules
            threshold_value = st_segment_analysis.get_threshold_for_lead_age_sex(
                lead, age, sex
            )

            # Compute baseline and threshold
            custom_baseline = st_segment_analysis.calculate_custom_baseline_value(signal)
            black_threshold = custom_baseline + threshold_value

            logger.info(f"custom_baseline (mV): {custom_baseline}")
            logger.info(f"threshold_value (mV): {threshold_value}")
            logger.info(f"black_threshold (mV): {black_threshold}")

            # Directory to store plots for this lead
            output_directory = f'{config.data.output_dir}/{csv_file_name}/{lead}'

            # Process each segment in this lead
            for i in range(num_segments):
                start_sample = i * samples_per_segment
                end_sample = start_sample + samples_per_segment
                end_sample = min(end_sample, len(signal))
                segment = signal[start_sample:end_sample]

                logger.info(
                    f"Segment {i+1}: Start {start_sample}, End {end_sample}, Length: {len(segment)}"
                )

                # Check and invert ECG segment if the raw signal is inverted, otherwise the signal is kept unchanged
                ecg_net_positive_seg, _ = nk.ecg_invert(segment, sampling_rate=config.plot.sampling_rate, show=False)

                # Detect R-peaks in this segment
                _, r_peaks = nk.ecg_peaks(
                    ecg_net_positive_seg,
                    sampling_rate=config.plot.sampling_rate,
                    method='martinez2004',
                    correct_artifacts=False,
                    show=False)

                r_peak_indices = r_peaks['ECG_R_Peaks']
                
                if len(r_peak_indices) == 0:
                    logger.info("No R-peaks detected for this segment; ECG delineation requires R-peaks. Skipping this segment.")
                    continue

                # Detect J-points from R-peak offsets (i.e. end of the QRS complexes)
                j_point = st_segment_analysis.detect_r_peak_offset(
                    ecg_net_positive_seg, r_peak_indices, sampling_rate=config.plot.sampling_rate
                )
                j_point = [x for x in j_point if not np.isnan(x)]  # Remove NaNs

                logger.info(f"J-point indices: {j_point}")
                logger.info(f"R-peak indices: {r_peak_indices}")

                # T-wave delineation
                try:
                    _, waves_dwt = nk.ecg_delineate(
                        ecg_net_positive_seg,
                        r_peak_indices,
                        sampling_rate=config.plot.sampling_rate,
                        method="dwt",
                        show=True,
                        show_type='bounds_T')
                except ValueError as e:
                    logger.info(f"Error delineating ECG: {e}")
                    continue

                # Extract T-wave onsets
                t_onsets = [int(t) for t in waves_dwt["ECG_T_Onsets"] if not np.isnan(t)]

                logger.info(f"T-wave onset indices: {t_onsets}")

                # Ensure matching count between J-points and T-onsets
                if len(j_point) != len(t_onsets):
                    logger.info("Error: The number of J-points and T-wave onsets must be the same.")
                    continue

                # Calculate ST interval in milliseconds (ms)
                st_ms = st_segment_analysis.calculate_time_ms_between_jpoint_and_twave(
                    j_point, t_onsets, sampling_rate=config.plot.sampling_rate
                )
                logger.info(f"ST (ms): {st_ms}")

                # Calculate STc using ST interval and RR interval
                stc = st_segment_analysis.calculate_stc(st_ms, rr_s)
                logger.info(f"STc (ms): {stc}")

                # Additional segment validation - check for NaN, zero or similar issues
                if np.all(np.isnan(segment)) or np.ptp(segment) == 0:
                    logger.info(
                        f"Segment {i+1} in lead {lead} contains only NaNs or zero amplitude. \
                            Skipping."
                    )
                    continue

                if len(segment) < samples_per_segment:
                    logger.info(
                        f"Segment {i+1} is shorter than expected in lead {lead}. Skipping."
                    )
                    continue

                segment_label=f"Segment {i+1}: Start {start_sample}, End {end_sample}, Length: {len(segment)}"

                # Call the plotting function for each segment
                st_elevation_plot.plot_ecg_on_red_mm_paper(
                    ecg_signal=segment,
                    j_points=j_point,
                    r_peaks=r_peak_indices,
                    custom_baseline=custom_baseline,
                    black_threshold=black_threshold,
                    sampling_rate=config.plot.sampling_rate,
                    seconds_to_display=config.plot.segment_duration_sec,
                    mm_per_second=config.plot.mm_per_second,
                    mm_per_mv=config.plot.mm_per_mv,
                    max_amplitude=max_amplitude,
                    output_dir=output_directory,
                    lead=lead,
                    segment=segment_label,
                    fig_title=csv_file_name,
                    show_custom_baseline=config.plot.show_custom_baseline,
                    show_black_threshold_line=config.plot.show_black_threshold_line,
                    fill_auc=config.plot.fill_auc,
                    show_axis_labels=config.plot.show_axis_labels,
                    show_inline_plot=config.plot.show_inline_plot,
                    occupy_full_fig_canvas=config.plot.occupy_full_fig_canvas,
                    save_plot=config.plot.save_plot,
                    show_r_peaks_marker=False,
                    show_jpoint_marker=True,
                    stc=stc,
                )


def main():
    """Load the config and run the pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        help="The path to the configuration file.",
    )
    args = parser.parse_args()

    settings = load_config(config_path=Path(args.config_path))

    run_pipeline(settings)


if __name__ == "__main__":
    main()
 