"""
ST-segment feature plotting functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Polygon
import logging
import re

from .analysis import get_threshold_for_lead_age_sex

logger = logging.getLogger(__name__)

def draw_horizontal_line(
    ax, y_value, color="black", linestyle="-", linewidth=0.3, label=None
):
    """
    Draw a horizontal line on the given axis at the specified y-value.

    Args:
    - ax: The axis on which to draw the horizontal line.
    - y_value: The y-coordinate of the horizontal line.
    - color (str): The color of the line. Default is 'black'.
    - linestyle (str): The style of the line. Default is '-' (solid line).
    - linewidth (float): the width of the black line drawn. Default is 0.3
    - label (str, optional): The label for the horizontal line. If specified, it will appear in
        the legend. Default is None

    Returns:
    - None
    """
    ax.axhline(y=y_value, color=color, linestyle=linestyle, lw=linewidth, label=label)


def plot_ecg_with_peaks(
    ecg_signals,
    r_peak_indices,
    r_offsets_dict,
    output_dir,
    fig_title,
    age,
    sex,
    leads=None,
):
    """
    Reader beware: Unused function. May need edits here or elsewhere to work.

    Plots ECG signals with R-peaks and R-offsets
    and saves each lead to a separate PNG file.

    Parameters:
    ecg_signals (dict): Dictionary containing ECG signals.
        Keys are lead identifiers and values are pandas Series of ECG signals.
    r_peak_indices (dict): Dictionary containing R-peak indices.
        Keys are lead identifiers and values are lists of R-peak indices.
    r_offsets_dict (dict): Dictionary containing R-offsets.
        Keys are lead identifiers and values are lists of R-offset indices.
    output_dir (str): The directory where the plots will be saved.
    fig_title (str): The title of the figure.
        Specifies the title of the plot.
    age (int): Age of the subject.
    sex (str): Sex of the subject ('M' or 'F').
    leads: List[str]
        An optional list of lead names to find R-peaks for.
        All leads will be used by default.

    Returns:
    None
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

    # Iterate through each lead to create and save individual plots
    for lead in leads:
        if lead not in ecg_signals:
            continue  # Skip any leads not present in ecg_signals

        signals = ecg_signals[lead]

        plt.figure(figsize=(36, 6))  # Create a new figure for each lead
        ax = plt.gca()  # Get current axis

        # Plot the ECG signal
        ax.plot(signals, label=f"ECG Lead {lead}")

        # Filter and plot valid R-peaks
        valid_r_peaks = [
            r_peak
            for r_peak in r_peak_indices.get(lead, [])
            if 0 <= r_peak < len(signals)
        ]
        ax.scatter(
            valid_r_peaks,
            signals.iloc[valid_r_peaks],
            color="red",
            label="R-peaks",
        )

        # Filter and plot valid R-offsets
        valid_r_offsets = [
            r_offset
            for r_offset in r_offsets_dict.get(lead, [])
            if 0 <= r_offset < len(signals)
        ]
        ax.scatter(
            valid_r_offsets,
            signals.iloc[valid_r_offsets],
            color="blue",
            label="R-offsets",
        )

        # Draw threshold line
        threshold = get_threshold_for_lead_age_sex(lead, age, sex)
        # draw black horizontal line
        draw_horizontal_line(ax, y_value=threshold, color="black")

        # Set the title and labels
        ax.set_title(f"ECG Signal with R-peaks and R-offsets for Lead {lead}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)  # Add grid lines

        # Add age and sex to the plot
        plt.text(
            0.01,
            0.95,
            f"Age: {age}, Sex: {sex}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.5},
        )

        # Save the plot with the original fig_title and lead name in the filename
        plt.tight_layout()  # Adjust layout to fit the title and labels
        plt.savefig(f"{output_dir}/{fig_title}_{lead}.png")

        plt.close()  # Close the figure to avoid memory leaks


def plot_ecg_on_red_mm_paper(
    ecg_signal,
    j_points,
    r_peaks,
    custom_baseline,
    black_threshold,
    sampling_rate=1000,
    seconds_to_display=5,
    mm_per_second=25,
    mm_per_mv=10,
    max_amplitude=None,
    output_dir=None,
    lead=None,
    segment=None,
    fig_title=None,
    show_custom_baseline=True,
    show_black_threshold_line=True,
    fill_auc=True,
    show_axis_labels=True,
    show_inline_plot=True,
    occupy_full_fig_canvas=True,
    save_plot=True,
    show_r_peaks_marker=True,
    show_jpoint_marker=True,
    stc=120
):
    """
    Plots an ECG signal on a red millimeter paper background with various customizable visual
    options.

    Parameters:
    - ecg_signal (pd.Series or np.array):
        The ECG signal to plot.
    - custom_baseline (float):
        A custom baseline value to overlay on the plot, often used to visualize signal
        alignment.
    - black_threshold (float):
        A threshold value to highlight specific parts of the signal, typically used for
        annotations or analysis.
    - sampling_rate (int, optional, default=1000):
        Sampling rate of the ECG signal in Hertz (Hz). Varies based on DB description
    - seconds_to_display (int, optional, default=5):
        Duration of the signal in seconds to display on the plot.
    - mm_per_second (int, optional, default=25):
        Speed of the ECG paper in millimeters per second.
    - mm_per_mv (int, optional, default=10):
        Scale in millimeters per millivolt.
    - max_amplitude (float, optional):
        Absolute maximum amplitude of the signal in millivolt (mV). Used to determine vertical extent of plot.
        If None, plot will be automatically fitted to the signal.
    - output_dir (str, optional):
        Directory path to save the plot as an image file.
    - lead (str, optional):
        Identifier for the ECG lead being plotted.
    - segment_number (int):
        Index of specific segment of the ECG signal.
    - fig_title (str, optional):
        Title of the plot for better identification.
    - show_custom_baseline (bool, optional, default=True):
        Whether to display the custom baseline on the plot.
    - show_black_threshold_line (bool, optional, default=True):
        Whether to display the black threshold line for additional signal analysis.
    - fill_auc (bool, optional, default=True):
        If `True`, fills the area under the curve (AUC) of the ECG signal.
    - show_axis_labels (bool, optional, default=True):
        Determines whether x-axis and y-axis labels should be shown on the plot.
    - show_inline_plot (bool, optional, default=True):
        If `True`, displays the plot inline within the Python environment (e.g., Jupyter Notebook).
    - occupy_full_fig_canvas (bool, optional, default=True):
        If `True`, make sure the generated plot is covering the full fig canvas and dont show any
        white padding.
    - save_plot (bool, optional, default=True):
        If 'True', save the plots

    Notes:
        If you want the saved figure to occupy the entire canvas without showing a
        black border line caused by figure labels set:
            occupy_full_fig_canvas = True
            show_axis_labels = False

    Returns:
    None
    """

    # Step 1: Check for errors in inputs
    if ecg_signal is None or len(ecg_signal) == 0:
        logger.info("Error: ECG signal is empty or None. Skipping this plot.")
        return

    if output_dir is None:
        logger.info("Error: No output directory specified. Skipping plot save.")
        return

    if lead is None or segment is None or fig_title is None:
        logger.info(
            "Error: Missing lead, segment, or figure title. Skipping this plot."
        )
        return

    # Step 2: Work out the min/max extent of the plot in the y-axis
    if max_amplitude is not None:
        y_min = -(max_amplitude)
        y_max = max_amplitude
    else:
        # Calculate amplitude range
        signal_min = ecg_signal.min()
        signal_max = ecg_signal.max()
        margin = 0.2  # Add a 0.2 mV margin for clarity
        y_min = signal_min - margin
        y_max = signal_max + margin

        # Round y_min and y_max to the nearest 1 mV step (to ensure it finishes on a large square)
        y_min = (
            np.floor((signal_min - margin) / 1) * 1
        )  # Round down to nearest multiple of 1 mV
        y_max = (
            np.ceil((signal_max + margin) / 1) * 1
        )  # Round up to nearest multiple of 1 mV

    # Step 3: Calculate figure size
    width_inches = (seconds_to_display * mm_per_second) / 25.4
    height_inches = (
        (y_max - y_min) * mm_per_mv
    ) / 25.4  # Adjust height based on signal amplitude to accurately fit signal to y-axis

    # Step 4: Plotting the ECG signal to resemble a red ECG millimeter paper
    fig, ax = plt.subplots(
        figsize=(width_inches, height_inches), dpi=500
    )
    ax.set_facecolor("#FFEBEB")  # Light red/pink colour for the background

    # Grid settings for 25 mm/sec
    large_squares = 0.2  # Each large square (200 ms) at 25 mm/sec
    small_squares = 0.04  # Each small square (40 ms) at 25 mm/sec

    # Major square lines (darker red for large squares)
    ax.grid(which="major", color="#F68D8D", linestyle="-", linewidth=0.3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(large_squares))
    ax.yaxis.set_major_locator(
        plt.MultipleLocator(0.5)
    )  # Adjust based on amplitude scale

    # Minor grid lines (lighter red for small squares)
    ax.grid(which="minor", color="#FFB6B6", linestyle="-", linewidth=0.15)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(small_squares))
    ax.yaxis.set_minor_locator(
        plt.MultipleLocator(0.1)
    )  # Adjust based on amplitude scale

    # Loop over segments and plot
    segment_length = seconds_to_display * sampling_rate  # Number of samples per segment

    # Determine the number of segments
    num_segments = len(ecg_signal) // segment_length

    for i in range(num_segments):
        # Extract the current segment
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        segment_data = ecg_signal[start_idx:end_idx]

        # Calculate time for this segment
        segment_time = np.linspace(
            i * seconds_to_display, (i + 1) * seconds_to_display, segment_length
        )

        # Plot ECG data for this segment
        ax.plot(segment_time, segment_data, color="black", linewidth=0.3)

        # Set axis limits for this segment
        ax.set_xlim(
            i * seconds_to_display, (i + 1) * seconds_to_display
        )  # Set x-limits for this segment
        ax.set_ylim(y_min, y_max)

        # Optionally plot baseline and threshold lines and AUC
        if fill_auc:
            # Convert to NumPy array to ensure consistent indexing
            above_threshold = (segment_data >= black_threshold).to_numpy()
        
            # Iterate through J-points and shade based on corresponding stc duration
            for j_index, j_point in enumerate(j_points):
                # Ensure j_point is valid
                if np.isnan(j_point) or not np.issubdtype(type(j_point), np.integer):
                    logger.info(f"Skipping invalid J-point: {j_point}")  # Debugging output
                    continue  # Skip invalid entries

                start_idx = int(j_point)  # Ensure it's an integer

                # Get corresponding duration in ms from stc and convert to samples
                duration_ms = stc[j_index]
                duration_samples = int((duration_ms / 1000) * sampling_rate)  # Convert ms to samples

                end_idx = min(start_idx + duration_samples, len(segment_data) - 1)  # Ensure within bounds
                logger.info("ST-segment start and end indices: %d to %d", start_idx, end_idx)

                # Detect continuous regions where the signal is above the threshold
                in_curve = False
                curve_start = None
                segments = []

                for i in range(start_idx, end_idx):
                    if above_threshold[i]:
                        if not in_curve:
                            in_curve = True
                            curve_start = i  # Start of the peak
                    else:
                        if in_curve:
                            in_curve = False
                            segments.append((curve_start, i - 1))  # Store only actual peak regions

                # If the signal ends while still in a peak, close the last segment
                if in_curve:
                    segments.append((curve_start, end_idx - 1))

                # Fill only the detected peaks
                for start, end in segments:
                    x_points = np.concatenate(([segment_time[start]], segment_time[start:end + 1], [segment_time[end]]))
                    y_points = np.concatenate(([black_threshold], segment_data[start:end + 1], [black_threshold]))
        
                    polygon = Polygon(
                        xy=list(zip(x_points, y_points)),
                        closed=True,
                        facecolor='black',
                        edgecolor='none',
                        alpha=1,
                        zorder=2
                    )
                    ax.add_patch(polygon)

        if show_jpoint_marker:
            for j_point in j_points:
                if np.isnan(j_point) or not np.issubdtype(type(j_point), np.integer):
                    continue
                start_idx = int(j_point)
                ax.scatter(segment_time[start_idx], segment_data.iloc[start_idx],
                               edgecolors='#3b979f', facecolors='none',
                               label="J-Point", zorder=5, s=5, linewidths=0.3)

        if show_custom_baseline:
            # draw horizontal line on custom detected baseline point
            draw_horizontal_line(
                ax,
                y_value=custom_baseline,
                color="orange",
                linestyle="-",
                linewidth=0.3,
                label="Baseline",
            )
        if show_black_threshold_line:
            # draw black horizontal line on the threshold point
            draw_horizontal_line(
                ax,
                y_value=black_threshold,
                color="black",
                linestyle="-",
                linewidth=0.3,
                label="Threshold",
            )

    # Optionally show x and y labels and numbers
    if show_axis_labels:
        # Set labels
        ax.set_ylabel("Amplitude (mV)", fontsize=5)
        ax.set_xticklabels([])

        # Add secondary x-axis with full second ticks
        secax = ax.secondary_xaxis("bottom")
        secax.set_xticks(range(0, int(segment_time[-1]) + 1))
        secax.set_xlabel("Time (s)", fontsize=5)
        secax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x)))
        for label in secax.get_xticklabels():
            label.set_fontsize(5)  # Adjust font size

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y}"))
        for label in ax.get_yticklabels():
            label.set_fontsize(5)  # Adjust font size

    else:
        # Hide major and minor tick labels
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # Remove plot borders by setting spine colors to white (effectively hides them)
        ax.tick_params(which="both", color="w")
        # Hide all tick marks by setting their color to white (or remove if preferred)
        ax.spines["top"].set_color("w")
        ax.spines["right"].set_color("w")
        ax.spines["bottom"].set_color("w")
        ax.spines["left"].set_color("w")

    # Split the 'fig_title' string at the '.csv' and keep the part before '.csv'
    fig_title_before_csv = fig_title.split(".csv")[0]


    # Use regular expression to search for the segment number in the 'segment' string
    extract_segment_number = re.search(r"Segment (\d+)", segment)

    # If a segment number is found, extract it
    if extract_segment_number:
        segment_number = extract_segment_number.group(1)  # Extract the segment number from the match

    if occupy_full_fig_canvas:
        # Set the Exact Bounds of the Plot Area to force the axes to occupy the full figure canvas.
        ax.set_position([0, 0, 1, 1])

    if show_r_peaks_marker:
        ax.scatter(segment_time[r_peaks], segment_data.iloc[r_peaks], color='purple', label="R-peaks", zorder=10, s=3)

    # Track errors during saving
    save_error_occurred = False

    if save_plot:
        try:
            # Ensure all intermediate directories are created
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory ensured: {output_dir}")

            # Construct filename for saving the plot
            filename = os.path.join(
                output_dir,
                f"{fig_title_before_csv}_{lead}_{segment_number}_{segment_length}.png",
            )

            # Save the plot
            plt.savefig(
                filename
            )  # Optionally add bbox_inches='tight', pad_inches=0 for precise layout
            logger.info(f"Plot successfully saved at: {filename}")
        except FileNotFoundError:
            logger.error(
                f"Error: Output directory '{output_dir}' does not exist and could not be created. \
                    Skipping plot save."
            )
            save_error_occurred = True
        except PermissionError as perm_error:
            logger.error(
                f"PermissionError: Cannot write to '{output_dir}'. Check permissions. \
                    Error: {perm_error}"
            )
            save_error_occurred = True
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving the plot: {e}")
            save_error_occurred = True

    # Show plot if enabled
    if show_inline_plot:
        try:
            plt.show()
            if save_error_occurred:
                logger.info("Note: Plot displayed inline despite save error.")
        except Exception as show_error:
            logger.error(f"An error occurred while displaying the plot: {show_error}")

    # Close the figure to avoid memory issues
    plt.close(fig)
