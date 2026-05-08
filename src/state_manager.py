import time
from collections import deque

class CognitiveBuffer:
    def __init__(self, duration_sec=1, fps_approx=3, cooldown_seconds=20):
        self.buffer = deque(maxlen=duration_sec * fps_approx)
                
        self.confusion_threshold = 0.6 
        
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0
        
    def add_state(self, state):
        if state:
            self.buffer.append(state)
            
    def requires_intervention(self):
        current_time = time.time()        
        
        if not hasattr(self, 'last_trigger_time'):
            self.last_trigger_time = 0
        self.cooldown_seconds = 20         
        
        if (current_time - self.last_trigger_time) < self.cooldown_seconds:           
            return False, None

        if len(self.buffer) < self.buffer.maxlen * 0.5:
            return False, None 
            
        confused_count = sum(1 for s in self.buffer if s in ['Confused', 'Frustrated'])
        ratio = confused_count / len(self.buffer)
                
        if ratio >= self.confusion_threshold:            
            print(f"🔥 FRICTION DETECTED! API Locked for {self.cooldown_seconds} seconds.")
            self.last_trigger_time = current_time
            self.buffer.clear()
            return True, "Confused/Frustrated"
            
        return False, None