import time

class CognitiveBuffer:
    def __init__(self, trigger_seconds=1.0, cooldown_seconds=20.0):
        self.trigger_seconds = trigger_seconds
        self.cooldown_seconds = cooldown_seconds        
      
        self.friction_start_time = None
        self.last_trigger_time = 0
        self.latest_state = "Neutral"
        
    def add_state(self, state):
        self.latest_state = state
        
    def requires_intervention(self):
        current_time = time.time()
        
        if not hasattr(self, 'friction_start_time'):
            self.friction_start_time = None
            self.last_trigger_time = 0
            self.trigger_seconds = 1.0
            self.cooldown_seconds = 20.0            
        
        if (current_time - self.last_trigger_time) < self.cooldown_seconds:
            self.friction_start_time = None 
            return False, None
       
        if self.latest_state in ['Confused', 'Frustrated']:            
            if self.friction_start_time is None:
                self.friction_start_time = current_time                
          
            if (current_time - self.friction_start_time) >= self.trigger_seconds:             
                self.last_trigger_time = current_time
                self.friction_start_time = None
                return True, self.latest_state
        else:           
            self.friction_start_time = None
            
        return False, None