def evaluate_peak_detection(detected, annotated, tolerance_samples):
    """Match detected peaks to annotations and calculate detection metrics."""
    detected = list(map(int, detected))
    annotated = list(map(int, annotated))
    used_detections = set()
    true_positives = 0
    timing_errors_samples = []

    for expected in annotated:
        candidates = [
            (abs(found - expected), index)
            for index, found in enumerate(detected)
            if index not in used_detections
            and abs(found - expected) <= tolerance_samples
        ]
        if candidates:
            timing_error, closest_index = min(candidates)
            used_detections.add(closest_index)
            true_positives += 1
            timing_errors_samples.append(timing_error)

    false_positives = len(detected) - true_positives
    false_negatives = len(annotated) - true_positives
    sensitivity = (
        true_positives / len(annotated) if annotated else 0.0
    )
    precision = true_positives / len(detected) if detected else 0.0
    mean_timing_error_samples = (
        sum(timing_errors_samples) / len(timing_errors_samples)
        if timing_errors_samples
        else 0.0
    )
    max_timing_error_samples = (
        max(timing_errors_samples) if timing_errors_samples else 0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "sensitivity": sensitivity,
        "precision": precision,
        "mean_timing_error_samples": mean_timing_error_samples,
        "max_timing_error_samples": max_timing_error_samples,
    }
