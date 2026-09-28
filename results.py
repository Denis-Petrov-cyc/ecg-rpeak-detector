import csv


def save_detection_results(file_path, peak_indices, peak_times, info):
    """Save detected beats and their derived measurements to CSV."""
    intervals = info.getRRintervals()
    heart_rates = info.getHeartRates()

    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["peak_number", "sample_index", "time_seconds", "rr_seconds", "bpm"]
        )
        for position, (sample_index, peak_time) in enumerate(
            zip(peak_indices, peak_times), start=1
        ):
            interval_index = position - 2
            rr_interval = "" if position == 1 else round(intervals[interval_index], 6)
            heart_rate = "" if position == 1 else round(heart_rates[interval_index], 3)
            writer.writerow(
                [position, int(sample_index), round(float(peak_time), 6), rr_interval, heart_rate]
            )
