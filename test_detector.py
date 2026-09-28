import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Infoextractor import Infoextractor
from RPeakDetector import RPeakDetector
from evaluation import evaluate_peak_detection
from results import save_detection_results
from signals import create_test_ecg, load_csv_ecg


class RPeakDetectorTests(unittest.TestCase):
    def test_detects_known_test_beats(self):
        signal, sampling_rate, expected_times = create_test_ecg(
            seconds=10, sampling_rate=250, heart_rate=60
        )
        _, detected_times = RPeakDetector(signal, sampling_rate).detectRPeaks()
        np.testing.assert_allclose(
            detected_times, expected_times, atol=1 / sampling_rate
        )

    def test_calculates_heart_rate(self):
        info = Infoextractor(np.array([0.5, 1.5, 2.5]))
        self.assertEqual(info.getRRintervals(), [1.0, 1.0])
        self.assertEqual(info.getHeartRates(), [60.0, 60.0])

    def test_loads_ecg_column_from_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecg.csv"
            path.write_text(
                "time_seconds,ecg_amplitude\n0.0,0.1\n0.1,1.2\n0.2,0.2\n",
                encoding="utf-8",
            )
            signal, sampling_rate = load_csv_ecg(path, 10)

        np.testing.assert_allclose(signal, [0.1, 1.2, 0.2])
        self.assertEqual(sampling_rate, 10)

    def test_exports_detection_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            peak_times = np.array([0.5, 1.5, 2.5])
            save_detection_results(
                path,
                np.array([125, 375, 625]),
                peak_times,
                Infoextractor(peak_times),
            )
            with path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["bpm"], "")
        self.assertEqual(rows[1]["bpm"], "60.0")

    def test_evaluates_peak_detection(self):
        metrics = evaluate_peak_detection(
            detected=[101, 199, 450],
            annotated=[100, 200, 300],
            tolerance_samples=5,
        )
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["mean_timing_error_samples"], 1.0)
        self.assertEqual(metrics["max_timing_error_samples"], 1)


if __name__ == "__main__":
    unittest.main()
