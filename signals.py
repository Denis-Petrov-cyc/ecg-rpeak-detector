import csv
import warnings
from pathlib import Path

import numpy as np


WFDB_BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q", "?",
}


def load_csv_ecg(file_path, sampling_rate, column="ecg_amplitude", seconds=None):
    """Load ECG amplitudes from a named column in a CSV file."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than zero")

    amplitudes = []
    with open(file_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV file must contain a header row")
        if column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Column '{column}' was not found. Available columns: {available}"
            )

        limit = None if seconds is None else int(seconds * sampling_rate)
        for row_number, row in enumerate(reader, start=2):
            if limit is not None and len(amplitudes) >= limit:
                break
            try:
                amplitudes.append(float(row[column]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid ECG value on CSV row {row_number}: {row[column]!r}"
                ) from error

    if not amplitudes:
        raise ValueError("CSV file contains no ECG samples")
    return np.asarray(amplitudes), sampling_rate


def load_physionet_ecg(
    record_name="100",
    seconds=10,
    database="mitdb",
    channel=0,
    data_directory=None,
):
    """Download an annotated ECG segment from PhysioNet using WFDB."""
    try:
        import wfdb
    except ImportError as error:
        raise ImportError(
            "WFDB is required. Install it with: python3 -m pip install wfdb"
        ) from error

    if data_directory:
        record_source = str(Path(data_directory) / record_name)
        physionet_directory = None
    else:
        record_source = record_name
        physionet_directory = database

    header = wfdb.rdheader(record_source, pn_dir=physionet_directory)
    sample_count = min(header.sig_len, int(seconds * header.fs))
    record = wfdb.rdrecord(
        record_source,
        pn_dir=physionet_directory,
        sampto=sample_count,
        channels=[channel],
    )
    annotations = wfdb.rdann(
        record_source,
        "atr",
        pn_dir=physionet_directory,
        sampto=sample_count,
    )

    beat_samples = np.asarray(
        [
            sample
            for sample, symbol in zip(annotations.sample, annotations.symbol)
            if symbol in WFDB_BEAT_SYMBOLS
        ],
        dtype=int,
    )
    signal = np.asarray(record.p_signal[:, 0])
    return signal, record.fs, beat_samples


def load_real_ecg(seconds=10):
    """Load a real recorded ECG segment sampled at 360 Hz."""
    sampling_rate = 360
    try:
        from scipy.datasets import electrocardiogram

        signal = electrocardiogram()
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scipy.misc import electrocardiogram

            signal = electrocardiogram()

    sample_count = min(len(signal), int(seconds * sampling_rate))
    return np.asarray(signal[:sample_count]), sampling_rate


def create_test_ecg(seconds=10, sampling_rate=250, heart_rate=60):
    """Create a repeatable ECG-like signal with known R-peak times."""
    time = np.arange(0, seconds, 1 / sampling_rate)
    signal = 0.03 * np.sin(2 * np.pi * 0.5 * time)
    beat_times = np.arange(0.5, seconds, 60 / heart_rate)

    for beat_time in beat_times:
        signal += 1.5 * np.exp(-0.5 * ((time - beat_time) / 0.015) ** 2)

    return signal, sampling_rate, beat_times
