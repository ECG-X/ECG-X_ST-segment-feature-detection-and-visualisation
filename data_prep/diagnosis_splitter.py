# -----------------------------------------------------------------------------------
# This module provides utility functions to extract and classify diagnosis information
# from .hea header files in PhysioNet-style ECG datasets.
#
# The classification currently supports splitting into "myocardial" and "control_group"
# based on the presence of a keyword ("myocardial" and "healthy control") in the "Reason for admission"
# line within the .hea file.
#
# These utilities are intended to support diagnosis-based folder organization but are
# optional and controlled via configuration (i.e., `separate_by_diagnosis: true/false`)
# to ensure compatibility across diverse PhysioNet datasets.

# +
import os
import logging

logger = logging.getLogger(__name__)


def classify_record(
    hea_path: str,
    diagnosis_keyword: str = "myocardial",
    healthy_keyword: str = "healthy control"
    ) -> str:
    """
    Classifies a patient record based on the diagnosis/reason in a .hea file 
    and determines the classification of the record into one of three categories:

        - "myocardial": if the diagnosis_keyword is found in the line.
        - "control_group": if the healthy_keyword is found in the line.
        - "other": if no match is found or the reason line is missing.

    Args:
        hea_path (str): Path to the .hea file to classify.
        diagnosis_keyword (str, optional): Keyword to detect myocardial cases.
            Defaults to "myocardial".
        healthy_keyword (str, optional): Keyword to detect healthy control cases.
            Defaults to "healthy control".

    Returns:
        str: The classification label: "myocardial", "control_group", or "other".
    """
    try:
        with open(hea_path, "r") as f:
            lines = f.readlines()

        # Find the 'reason for admission' line
        reason_line = None
        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith("# reason for admission:"):
                reason_line = stripped.split(":", 1)[-1]
                break

        # Classify based on keywords in the 'reason for admission' line
        if diagnosis_keyword.lower() in reason_line:
            label = "myocardial"
        elif healthy_keyword.lower() in reason_line:
            label = "control_group"
        else:
            label = "other"

        logger.info(
            f"Class: {label} | "
            f"File: {os.path.basename(hea_path)} | "
            f"Diagnosis: {reason_line if reason_line else 'Unknown'}"
        )

        return label

    except Exception as e:
        logger.warning(f"Failed to classify {hea_path}: {e}")
        return "other"
