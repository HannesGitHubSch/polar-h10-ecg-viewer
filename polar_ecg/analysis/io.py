"""
Unified recording file reader supporting legacy CSV v1 and modern CSV v2 with metadata.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Recording:
    """
    Standardized in-memory representation of an ECG recording session.
    """

    df: pd.DataFrame
    format_version: int
    participant_id: str
    meta: dict
    hr_df: pd.DataFrame | None = None
    beats_df: pd.DataFrame | None = None


def load_recording(csv_path: str | Path) -> Recording:
    """
    Load an ECG recording from CSV, detecting format version (v1 or v2).
    Returns a unified Recording object with standardized column names:
    ['sample_index', 't_session_s', 'ecg_uv', 'host_time_utc', 'gap_before', 'event'].
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Recording file not found: {path}")

    # Read header to detect format
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    header_cols = [c.strip() for c in first_line.split(",")]

    if "sample_index" in header_cols:
        # Format v2
        df = pd.read_csv(path)
        format_version = 2

        # Load sidecar metadata if available
        meta_path = path.with_name(f"{path.stem}_meta.json")
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not read sidecar metadata %s: %s", meta_path, e)

        participant_id = meta.get("participant_id", "")
        if not participant_id:
            # Fallback parse from filename e.g. ECG_P001_...
            parts = path.stem.split("_")
            participant_id = parts[1] if len(parts) >= 2 else "Unknown"

        # Load auxiliary beats if present
        beats_path = path.with_name(f"{path.stem}_beats.csv")
        beats_df = pd.read_csv(beats_path) if beats_path.exists() else None

        # Load auxiliary HR if present
        hr_path = path.with_name(f"{path.stem}_hr.csv")
        hr_df = pd.read_csv(hr_path) if hr_path.exists() else None

        # Fill NaN events with empty string
        if "event" in df.columns:
            df["event"] = df["event"].fillna("")

        return Recording(
            df=df,
            format_version=format_version,
            participant_id=participant_id,
            meta=meta,
            hr_df=hr_df,
            beats_df=beats_df,
        )

    elif "participant_id" in header_cols and "ecg_voltage_uv" in header_cols:
        # Legacy Format v1
        logger.warning(
            "Loading legacy CSV format v1 from %s. Timestamps are host-derived arrival times.",
            path.name,
        )
        raw_df = pd.read_csv(path)
        format_version = 1
        n = len(raw_df)

        participant_id = str(raw_df["participant_id"].iloc[0]) if n > 0 else "Unknown"

        # Construct normalized DataFrame
        fs = 130.0
        sample_index = np.arange(n, dtype=int)
        t_session_s = sample_index / fs

        norm_df = pd.DataFrame({
            "sample_index": sample_index,
            "t_session_s": t_session_s,
            "ecg_uv": raw_df["ecg_voltage_uv"].values.astype(float),
            "host_time_utc": raw_df["sample_datetime"].values,
            "gap_before": np.zeros(n, dtype=int),
            "event": ["" for _ in range(n)],
        })

        meta = {
            "format": "legacy-polar-ecg-v1",
            "format_version": 1,
            "participant_id": participant_id,
            "fs_nominal": fs,
            "samples_written": n,
        }

        return Recording(
            df=norm_df,
            format_version=format_version,
            participant_id=participant_id,
            meta=meta,
        )

    else:
        raise ValueError(f"Unrecognized CSV header format in {path}: {first_line}")
