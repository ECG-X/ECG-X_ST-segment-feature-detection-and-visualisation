# Unit tests for both st_segment and shared folders in pipeline excluding plot functions

import sys
import os
import math
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

# Get the current working directory
base_dir = os.getcwd()
print(f"Current working directory: {base_dir}")

# Construct the path to the project root
project_root = os.path.abspath(os.path.join(base_dir, '..'))
print(f"Adding {project_root} to sys.path")

if project_root not in sys.path:
    sys.path.insert(0, project_root)  # insert at front to take priority

# import modules from pipeline/
try:
    from pipeline.shared.data_processing import (
        generate_ecg_time_column,
        read_12_lead_ecg_data
    )
    from pipeline.shared.signal_processing import (
        correct_baseline_wandering,
        remove_noise,
        read_and_invert_12_lead_ecg_data
    )
    from pipeline.st_segment.analysis import (
        find_r_peaks,
        calculate_averaged_rr_interval_for_leads,
        calculate_heart_rate,
        calculate_time_ms_between_jpoint_and_twave,
        calculate_stc,
        detect_r_peak_offset,
        get_threshold_for_lead_age_sex,
        calculate_custom_baseline_value
    )
    from pipeline.st_segment.plot import (
        draw_horizontal_line,
    )
    print("Modules imported successfully.")
except ModuleNotFoundError as e:
    print(f"Error importing module: {e}")


class TestECGFunctions(unittest.TestCase):

    # ----------------------------
    # Helper Assertions
    # ----------------------------
    def assertFloatAlmostEqual(self, expected, actual, relative_tolerance):
        """Assert two floats are approximately equal given a relative tolerance."""
        absolute_tolerance = abs(expected) * relative_tolerance
        self.assertAlmostEqual(expected, actual, delta=absolute_tolerance)

    def _assert_ecg_dataframe_valid(self, df, expected_leads):
        """
        Validate ECG DataFrame:
        - Must contain requested leads + "time"
        - Must have the same number of rows as self.ECG_time
        """
        expected_columns = expected_leads + ["time"]
        self.assertCountEqual(df.columns.tolist(), expected_columns)
        self.assertEqual(len(df), len(self.ECG_time))

    def _assert_dataframe_same_shape_and_columns(self, df, reference_df):
        """
        Assert two DataFrames have:
        - The same shape (#rows, #cols)
        - Identical column names
        """
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, reference_df.shape)
        self.assertListEqual(list(df.columns), list(reference_df.columns))

    def _assert_series_almost_equal(self, original, processed, tolerance=1e-6, less_than_equal=True):
        """
        Compare two pandas Series element-wise:
        - By default, checks that processed values are <= original values (within tolerance).
        - If less_than_equal=False, checks numerical closeness (rtol).
        """
        if less_than_equal:
            for o, p in zip(original, processed):
                self.assertLessEqual(abs(p), abs(o) + tolerance)
        else:
            pd.testing.assert_series_equal(original, processed, check_exact=False, rtol=tolerance)

    # ----------------------------
    # Helper Data Generation
    # ----------------------------
    def _generate_synthetic_ecg(self, num_samples=2000, sampling_rate=500, include_noise=True, include_high_freq=False):
        """
        Create synthetic ECG signals for testing:
        - base_ecg: 5 Hz sine wave
        - drift: 0.2 Hz slow baseline wander
        - noise: Gaussian noise (optional)
        - high_freq_noise: 100 Hz noise (optional)
        """
        time = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)
        base_ecg = np.sin(2 * np.pi * 5 * time)
        drift = 0.5 * np.sin(2 * np.pi * 0.2 * time)
        noise = 0.05 * np.random.randn(num_samples) if include_noise else 0
        high_freq_noise = 0.3 * np.sin(2 * np.pi * 100 * time) if include_high_freq else 0

        ecg_mv = pd.DataFrame({
            lead: base_ecg + drift + noise + high_freq_noise
            for lead in ["lead_I", "lead_II", "lead_III"]
        })
        return ecg_mv, time

    def _generate_time_column(self, num_samples=None, sampling_rate=None):
        """
        Generate ECG time column using project’s function.
        Falls back to default class attributes if not provided.
        """
        num_samples = num_samples or self.num_samples
        sampling_rate = sampling_rate or self.sampling_rate
        return generate_ecg_time_column(sampling_rate=sampling_rate, num_samples=num_samples)

    # ----------------------------
    # Setup
    # ----------------------------
    def setUp(self):
        """Runs before every test: generate default ECG data and metadata."""
        np.random.seed(0)
        self.num_samples = 30000
        self.num_leads = 12
        self.sampling_rate = 1000
        self.custom_leads = ["i", "ii", "iii"] 

        self.ECG_time = self._generate_time_column()

        self.ECG_mv = pd.DataFrame(
            np.random.randn(self.num_samples, self.num_leads),
            columns=["i", "ii", "iii", "avr", "avl", "avf",
                     "v1", "v2", "v3", "v4", "v5", "v6"]
        )
        self.ECG_mv["Age"] = 55
        self.ECG_mv["Sex"] = "M"

        # Common RR peaks test data
        self.r_peaks_times = {
            "i": [0, 800, 1600, 2400],
            "ii": [0, 1000, 2000, 3100],
            "iii": [0, 50, 100]  # invalid intervals
        }
    
        # Expected average RR intervals (some valid, some NaN)
        self.rr_values_dict = {
            "i": 1,
            "ii": 0.8,
            "iii": np.nan
        }
        self.synthetic_ecg, self.synthetic_time = self._generate_synthetic_ecg()

    # ----------------------------
    # Tests Shared
    # ----------------------------
    def test_generate_ecg_time_column(self):
        """Check that ECG time column is generated correctly."""
        ECG_time_custom = self._generate_time_column()
        self.assertIsInstance(ECG_time_custom, pd.DataFrame)
        self.assertEqual(ECG_time_custom.columns.tolist(), ["s"])
        self.assertEqual(len(ECG_time_custom), self.num_samples)
        
        # Final time value should be close to num_samples / sampling_rate
        expected_value = self.num_samples / self.sampling_rate
        actual_value = ECG_time_custom["s"].iloc[-1]
        self.assertFloatAlmostEqual(expected_value, actual_value, relative_tolerance=0.001)

    def test_read_12_lead_ecg_data(self):
        """Check reading ECG data returns requested leads + metadata."""

        # Mock CSV read to return synthetic ECG
        with patch("pandas.read_csv", return_value=self.ECG_mv):
            ECG, age, sex = read_12_lead_ecg_data("fake_path.csv", self.ECG_time, leads=self.custom_leads)
        # Validate output structure + metadata
        self._assert_ecg_dataframe_valid(ECG, self.custom_leads)
        self.assertEqual(age, 55)
        self.assertEqual(sex, "M")

    def test_correct_baseline_wandering(self):
        """Check that baseline correction reduces mean offset in ECG signals."""
        ecg_mv, time = self._generate_synthetic_ecg()
        corrected = correct_baseline_wandering(ecg_mv.copy(), time, sampling_rate=500, plot_result=False)

        self._assert_dataframe_same_shape_and_columns(corrected, ecg_mv)
        self._assert_series_almost_equal(ecg_mv.mean(), corrected.mean())

    def test_remove_noise(self):
        """Check that noise removal reduces variance and mean drift."""
        ecg_mv, time = self._generate_synthetic_ecg(include_high_freq=True)
        filtered = remove_noise(ecg_mv.copy(), time, sampling_rate=500, plot_result=False)

        self._assert_dataframe_same_shape_and_columns(filtered, ecg_mv)
        self._assert_series_almost_equal(ecg_mv.var(), filtered.var(), less_than_equal=True)
        self._assert_series_almost_equal(ecg_mv.mean(), filtered.mean())

    def test_read_and_invert_12_lead_ecg_data_success(self):
        """Check inversion of ECG signals using neurokit2.ecg_invert."""
        # Mock inversion to simply negate the signal
        with patch("neurokit2.ecg_invert", side_effect=lambda sig, *a, **k: (-sig, None)):
            inverted_ECG = read_and_invert_12_lead_ecg_data(
                self.ECG_mv, self.ECG_time, sampling_rate=self.sampling_rate, leads=self.custom_leads
            )

        self._assert_ecg_dataframe_valid(inverted_ECG, self.custom_leads)
        # Each lead should be inverted compared to original
        for lead in self.custom_leads:
            pd.testing.assert_series_equal(
                inverted_ECG[lead].reset_index(drop=True),
                -self.ECG_mv[lead].reset_index(drop=True),
                check_names=False
            )

    def test_read_and_invert_12_lead_ecg_data_missing_leads(self):
        """Check error is raised when requested lead is missing."""
        with self.assertRaises(KeyError):
            read_and_invert_12_lead_ecg_data(
                self.ECG_mv.drop(columns=["i"]), self.ECG_time, sampling_rate=self.sampling_rate, leads=["i"]
            )

    def test_read_and_invert_12_lead_ecg_data_missing_time_column(self):
        """Check error is raised when time column is missing."""
        bad_time = self.ECG_time.rename(columns={"s": "timestamp"})
        with self.assertRaises(KeyError):
            read_and_invert_12_lead_ecg_data(self.ECG_mv, bad_time, sampling_rate=self.sampling_rate, leads=["i"])

    # ----------------------------
    # Tests ST-segment analysis
    # ----------------------------
    def test_find_r_peaks(self):
        """Check that R-peaks are detected correctly using mocked neurokit2.ecg_peaks."""

        # Fake R-peak indices
        fake_peaks = np.array([10, 50, 100])

        # Patch nk.ecg_peaks to return controlled output
        with patch("neurokit2.ecg_peaks", return_value=(
            {}, {"ECG_R_Peaks": fake_peaks, "sampling_rate": self.sampling_rate})) as mock_peaks:
            peaks = find_r_peaks(self.ECG_mv, leads=self.custom_leads, sampling_rate=self.sampling_rate)

        # Validate peaks dictionary contains expected leads and values
        self.assertIsInstance(peaks, dict)
        for lead in self.custom_leads:
            self.assertIn(lead, peaks)
            expected_peaks =  fake_peaks / self.sampling_rate * 1000
            self.assertEqual(list(peaks[lead]), list(expected_peaks))

        # Ensure ecg_peaks was called once per lead
        self.assertEqual(mock_peaks.call_count, len(self.custom_leads))

    def test_calculate_averaged_rr_interval_for_leads(self):
        averages = calculate_averaged_rr_interval_for_leads(self.r_peaks_times)
        self.assertAlmostEqual(averages["i"], 0.8, delta=1e-6)
        self.assertAlmostEqual(averages["ii"], 1.0333333, delta=1e-3)
        self.assertTrue(np.isnan(averages["iii"]))
    
    def test_calculate_heart_rate(self):
        heart_rates = calculate_heart_rate(self.rr_values_dict)
        self.assertAlmostEqual(heart_rates["i"], 60.0, delta=1e-6)
        self.assertAlmostEqual(heart_rates["ii"], 75.0, delta=1e-6)
        self.assertTrue(np.isnan(heart_rates["iii"]))

    def test_calculate_time_ms_between_jpoint_and_twave(self):
        # Normal case: positive differences
        j_points = [100, 200, 300]
        t_onsets = [150, 250, 350]  # 50 samples difference
        expected = [50.0, 50.0, 50.0]
    
        result = calculate_time_ms_between_jpoint_and_twave(j_points, t_onsets, sampling_rate=self.sampling_rate)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, delta=1e-6)
    
        # Edge case: zero difference
        j_points = [100, 200]
        t_onsets = [100, 200]
        expected = [0.0, 0.0]
    
        result = calculate_time_ms_between_jpoint_and_twave(j_points, t_onsets, sampling_rate=self.sampling_rate)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, delta=1e-6)
    
        # Error case: mismatched lengths → should return None
        j_points = [100, 200]
        t_onsets = [150]  # shorter
        with self.assertRaises(ValueError):
            calculate_time_ms_between_jpoint_and_twave(j_points, t_onsets, sampling_rate=self.sampling_rate)
        

    def test_calculate_stc(self):
        # Normal case
        st_ms = [100, 120, 140]  # milliseconds
        rr_interval = 1.0  # seconds (60 bpm)
        result = calculate_stc(st_ms, rr_interval)
        expected = [100.0, 120.0, 140.0]  # sqrt(1) = 1  unchanged
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, delta=1e-6)
    
        # Case with different RR interval
        st_ms = [100, 200]
        rr_interval = 0.25  # seconds (240 bpm)
        result = calculate_stc(st_ms, rr_interval)
        expected = [100 / math.sqrt(0.25), 200 / math.sqrt(0.25)]  # sqrt(0.25) = 0.5
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, delta=1e-6)
    
        # Error case: invalid rr_interval
        with self.assertRaises(ValueError):
            calculate_stc([100, 120], 0)
    
        with self.assertRaises(ValueError):
            calculate_stc([100, 120], -1)

    def test_detect_r_peak_offset(self):
        # Patch nk.ecg_delineate to simulate successful delineation
        mock_waves = {"ECG_R_Offsets": [120, 220, 320]}
        with patch("neurokit2.ecg_delineate", return_value=(None, mock_waves)):
            r_offsets = detect_r_peak_offset(self.synthetic_ecg, [100, 200, 300], sampling_rate=self.sampling_rate)
    
        self.assertEqual(r_offsets, [120, 220, 320])

    def test_get_threshold_for_lead_age_sex(self):
        # Default case (non-V2/V3 leads)
        self.assertEqual(get_threshold_for_lead_age_sex("i", 30, "male"), 0.1)
        # Male, V2 lead, age < 40
        self.assertEqual(get_threshold_for_lead_age_sex("v2", 25, "male"), 0.25)
        # Male, V3 lead, age >= 40
        self.assertEqual(get_threshold_for_lead_age_sex("v3", 45, "male"), 0.2)
        # Female, V2 lead
        self.assertEqual(get_threshold_for_lead_age_sex("v2", 50, "female"), 0.15)
        # Female, V3 lead
        self.assertEqual(get_threshold_for_lead_age_sex("v3", 35, "female"), 0.15)

    def test_calculate_custom_baseline_value(self):
        signal = self.synthetic_ecg["lead_I"]
        baseline = calculate_custom_baseline_value(signal)
        # Validate baseline type
        self.assertIsInstance(baseline, float)

    def test_draw_horizontal_line(self):
        fig, ax = plt.subplots()
        draw_horizontal_line(ax, y_value=0.5, color="red", linestyle="--", linewidth=1.5, label="TestLine")
        lines = ax.get_lines()
        self.assertTrue(any(line.get_ydata()[0] == 0.5 for line in lines))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
