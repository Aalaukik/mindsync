import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

class MindSyncOrchestrator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')         
    def generate_nudge(self, emotion, topic, current_content):       
        system_prompt = f"""
        You are the MindSync Mentor. You have noticed the student is {emotion} while studying {topic}. 
        Do not be overbearing. Briefly offer a 'Mental Reset'-this could be a simplified analogy, a quick hint, or a gentle nudge to refocus. 
        Your goal is to restore their 'Flow State' in under 50 words.
        Current material context: "{current_content}"
        """
        
        response = self.model.generate_content(system_prompt)
        return response.text