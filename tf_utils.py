class LinearSchedule():
    def __init__(self, start=1.0, steps=10000, stop=0.02):
        self.func = lambda t: stop + (start - stop) * (float(max(0, steps - t)) / steps)

    def eps(self, t):
        return self.func(t)
