"""
Configuration schema & utility methods
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel
import yaml


class DataSettings(BaseModel):
    """Settings specific to input/output data"""

    input_dir: str
    output_dir: str
    leads: List[str]


class PlotSettings(BaseModel):
    """Settings specific to plots"""

    sampling_rate: int
    segment_duration_sec: int
    total_duration_sec: int
    mm_per_second: int
    mm_per_mv: int
    max_amplitude: int
    show_custom_baseline: bool
    show_black_threshold_line: bool
    fill_auc: bool
    show_axis_labels: bool
    show_inline_plot: bool
    occupy_full_fig_canvas: bool
    save_plot: bool 


class DebugSettings(BaseModel):
    """Settings specific to debugging"""

    plot_intermediate_ecgs: bool


class Settings(BaseModel):
    """The configuration settings"""
    
    data: DataSettings
    plot: PlotSettings
    debug: DebugSettings


def load_config(config_path: Path) -> Settings:
    """Helper function to load the configuration yaml file.

    Args:
        config: pth to the the configuration file.

    Returns:
        the configuration settings instantiated as a Settings object.
    """
    with open(config_path, mode="r") as in_file:
        settings = yaml.safe_load(in_file)

    return Settings(**settings)
