import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

class MindSyncOrchestrator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)            
        
        system_instruction = "You are an empathetic AI tutor. Provide a single, short, empathetic sentence (maximum 10 words) to help the user reset and refocus. Do not use quotes."
            
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_instruction,           
            generation_config=genai.GenerationConfig(
                max_output_tokens=25, 
                temperature=0.4,     
            )
        )

    def generate_nudge(self, emotion, topic, current_content):       
        prompt = f"State: {emotion}. Topic: {topic}."
        
        try:            
            response = self.model.generate_content(
                prompt, 
                request_options={"timeout": 2.0} 
            )
            return response.text.strip()
            
        except ResourceExhausted:
            return "Take a deep breath and stretch. You've got this!"
        except Exception as e:            
            return "Let's pause for a moment and refocus."