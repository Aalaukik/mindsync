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
        
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_nudge(self, emotion, topic, current_content):
        system_prompt = f"""
        You are an empathetic AI tutor. 
        The user is currently experiencing: {emotion}.
        They are working on: {topic}.
        Current context: {current_content}.
        
        Provide a single, short, empathetic sentence to help them reset and refocus. Do not use quotes.
        """
        
        try:            
            response = self.model.generate_content(system_prompt)
            return response.text.strip()
            
        except ResourceExhausted:            
            print("API Limit Hit: Serving fallback nudge.")
            return "Take a deep breath and stretch. You've got this! (API cooling down...)"
            
        except Exception as e:           
            print(f"API Error: {e}")
            return "Let's pause for a moment and refocus."