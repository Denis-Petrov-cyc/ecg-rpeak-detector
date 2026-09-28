import argparse

import matplotlib.pyplot as plt

from Infoextractor import Infoextractor
from RPeakDetector import RPeakDetector
from evaluation import evaluate_peak_detection
from results import save_detection_results
from signals import (
    create_test_ecg,
    load_csv_ecg,
    load_physionet_ecg,
    load_real_ecg,
)


def main():
    parser = argparse.ArgumentParser(description="Detect R peaks in ECG data")
    parser.add_argument(
        "--source",
        choices=("real", "test", "csv", "physionet"),
        default="real",
        help="use a recorded ECG or a generated test signal",
    )
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--file", help="path to a CSV file when --source csv is used")
    parser.add_argument(
        "--sampling-rate",
        type=float,
        help="CSV sampling rate in samples per second",
    )
    parser.add_argument(
        "--column",
        default="ecg_amplitude",
        help="CSV column containing ECG amplitudes",
    )
    parser.add_argument("--output", help="save detected heartbeat results to CSV")
    parser.add_argument(
        "--record",
        default="100",
        help="PhysioNet record name (default: 100)",
    )
    parser.add_argument(
        "--database",
        default="mitdb",
        help="PhysioNet database name (default: mitdb)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="ECG channel number (default: 0)",
    )
    parser.add_argument(
        "--data-dir",
        help="directory containing downloaded PhysioNet record files",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")

    if args.source == "real":
        ekgSignal, samplingRate = load_real_ecg(args.seconds)
    elif args.source == "test":
        ekgSignal, samplingRate, expectedPeaks = create_test_ecg(args.seconds)
        print("Expected test peak times:", expectedPeaks)
    elif args.source == "csv":
        if not args.file:
            parser.error("--file is required when --source csv is used")
        if args.sampling_rate is None:
            parser.error("--sampling-rate is required when --source csv is used")
        try:
            ekgSignal, samplingRate = load_csv_ecg(
                args.file,
                args.sampling_rate,
                column=args.column,
                seconds=args.seconds,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        try:
            ekgSignal, samplingRate, annotatedPeaks = load_physionet_ecg(
                record_name=args.record,
                seconds=args.seconds,
                database=args.database,
                channel=args.channel,
                data_directory=args.data_dir,
            )
        except Exception as error:
            parser.error(f"Could not load PhysioNet record: {error}")

    peakIndices, rPeaks = RPeakDetector(
        ekgSignal, samplingRate
    ).detectRPeaks()
    info = Infoextractor(rPeaks)

    print("Peak indices:", peakIndices)
    print("R-peak times (seconds):", rPeaks)
    print(
        "RR intervals (seconds):",
        [float(round(interval, 2)) for interval in info.getRRintervals()],
    )
    print(
        "Heart rates (BPM):",
        [float(round(heartRate, 2)) for heartRate in info.getHeartRates()],
    )
    if len(rPeaks) >= 2:
        print("Average heart rate:", round(info.getAverageHeartRate(), 2), "BPM")
    if args.source == "physionet":
        print("Annotated peak indices:", annotatedPeaks)
        print("Detected peaks:", len(peakIndices))
        print("Annotated beats:", len(annotatedPeaks))
        metrics = evaluate_peak_detection(
            peakIndices,
            annotatedPeaks,
            tolerance_samples=round(0.1 * samplingRate),
        )
        print("Correct detections:", metrics["true_positives"])
        print("False detections:", metrics["false_positives"])
        print("Missed beats:", metrics["false_negatives"])
        print("Sensitivity:", f'{metrics["sensitivity"]:.1%}')
        print("Precision:", f'{metrics["precision"]:.1%}')
        milliseconds_per_sample = 1000 / samplingRate
        print(
            "Mean timing error:",
            f'{metrics["mean_timing_error_samples"] * milliseconds_per_sample:.2f} ms',
        )
        print(
            "Maximum timing error:",
            f'{metrics["max_timing_error_samples"] * milliseconds_per_sample:.2f} ms',
        )

    if args.output:
        try:
            save_detection_results(args.output, peakIndices, rPeaks, info)
        except OSError as error:
            parser.error(f"Could not save results: {error}")
        print("Results saved to:", args.output)

    if not args.no_plot:
        time = [i / samplingRate for i in range(len(ekgSignal))]
        plt.plot(time, ekgSignal, label="ECG")
        plt.scatter(
            rPeaks,
            [ekgSignal[i] for i in peakIndices],
            color="red",
            label="R peaks",
        )
        if args.source == "physionet":
            annotatedTimes = annotatedPeaks / samplingRate
            plt.scatter(
                annotatedTimes,
                [ekgSignal[i] for i in annotatedPeaks],
                color="green",
                marker="x",
                label="PhysioNet annotations",
            )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title(f"R-peak detection ({args.source} signal)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
