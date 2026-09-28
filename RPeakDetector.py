import numpy as np
from scipy.signal import find_peaks


class RPeakDetector:
    def __init__(self, ekgSignal, samplingRate):
        self.ekgSignal = np.asarray(ekgSignal, dtype=float)
        self.samplingRate = samplingRate

    def detectRPeaks(self):
        if self.samplingRate <= 0:
            raise ValueError("samplingRate must be greater than zero")
        if self.ekgSignal.size < 3:
            return np.array([], dtype=int), np.array([], dtype=float)

        peaks, _ = find_peaks(
            self.ekgSignal,
            distance=max(1, int(0.3 * self.samplingRate)),
            prominence=0.5,
        )
        peakTimes = peaks / self.samplingRate
        return peaks, peakTimes
