
class HMMState:
    fromTimestamp = None
    label = None
    duration = -1
    probability = -1
    hiddenstate = None
    
    def __init__(self, timestamp, lbl, dur):
        self.fromTimestamp = timestamp
        self.label = lbl
        self.duration = dur
        
    def toString(self):
        hmmStr = "%s , %s , %s , %s , %f" % (str(self.fromTimestamp), self.label, str(self.duration), str(self.hiddenstate), (self.probability))
        return(hmmStr)