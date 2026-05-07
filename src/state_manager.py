import time
from collections import deque

class CognitiveBuffer:
    def __init__(self, duration_sec=30, fps_approx=10):
        self.buffer = deque(maxlen=duration_sec * fps_approx)
        self.confusion_threshold = 0.6 # 60% of recent frames show confusion/frustration
        
    def add_state(self, state):
        if state:
            self.buffer.append(state)
            
    def requires_intervention(self):
        if len(self.buffer) < self.buffer.maxlen * 0.5:
            return False, None # Not enough data yet
            
        confused_count = sum(1 for s in self.buffer if s in ['Confused', 'Frustrated'])
        ratio = confused_count / len(self.buffer)
        
        if ratio >= self.confusion_threshold:
            # Clear buffer to prevent spamming nudges
            self.buffer.clear()
            return True, "Confused/Frustrated"
        return False, None