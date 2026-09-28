class Infoextractor:
    def __init__(self, Rpeaks):
        self.Rpeaks = Rpeaks

    def getRRintervals(self):
        intervals = []
        for i in range(len(self.Rpeaks) - 1):
            difference = self.Rpeaks[i + 1] - self.Rpeaks[i]
            intervals.append(difference)
        return intervals

    def getHeartRates(self):
        intervals = self.getRRintervals()
        heartRates = []
        for interval in intervals:
            if interval <= 0:
                raise ValueError("R-peak times must be in increasing order")
            bpm = 60 / interval
            heartRates.append(bpm)
        return heartRates

    def getAverageHeartRate(self):
        heartRates = self.getHeartRates()
        if not heartRates:
            raise ValueError("At least two R peaks are required")
        return sum(heartRates) / len(heartRates)

    def getMinHeartRate(self):
        heartRates = self.getHeartRates()
        if not heartRates:
            raise ValueError("At least two R peaks are required")
        return min(heartRates)

    def getMaxHeartRate(self):
        heartRates = self.getHeartRates()
        if not heartRates:
            raise ValueError("At least two R peaks are required")
        return max(heartRates)
