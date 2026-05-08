import time

class CognitiveBuffer:
    def __init__(self, duration_sec=1.0, cooldown_seconds=20):
        self.duration_sec = duration_sec
        self.confusion_threshold = 0.6 
        self.cooldown_seconds = cooldown_seconds
                
        self.state_history = [] 
        self.last_trigger_time = 0
        
    def add_state(self, state):
        if state:
            current_time = time.time()
            self.state_history.append((current_time, state))
                        
            cutoff_time = current_time - self.duration_sec
            self.state_history = [s for s in self.state_history if s[0] >= cutoff_time]
            
    def requires_intervention(self):
        current_time = time.time()
                
        if not isinstance(self.state_history, list):
            self.state_history = []
            self.last_trigger_time = 0
            self.cooldown_seconds = 20        
        
        if (current_time - self.last_trigger_time) < self.cooldown_seconds:
            return False, None
        
        if len(self.state_history) < 2:
            return False, None 
                  
        confused_count = sum(1 for timestamp, state in self.state_history if state in ['Confused', 'Frustrated'])
        ratio = confused_count / len(self.state_history)
                
        if ratio >= self.confusion_threshold:            
            print(f"🔥 FRICTION DETECTED! API Locked for {self.cooldown_seconds} seconds.")
            self.last_trigger_time = current_time
            self.state_history.clear()            
            
            latest_emotion = "Frustrated" if confused_count > 0 else "Confused"
            return True, latest_emotion
            
        return False, None