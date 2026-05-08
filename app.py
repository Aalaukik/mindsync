import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import time
from streamlit_option_menu import option_menu
from src.cv_pipeline import EmotionClassifier
from src.state_manager import CognitiveBuffer
from src.llm_orchestrator import MindSyncOrchestrator

if "buffer" not in st.session_state:
    st.session_state.buffer = CognitiveBuffer()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MindSyncOrchestrator()
if "current_nudge" not in st.session_state:
    st.session_state.current_nudge = ""

st.set_page_config(layout="wide", page_title="MindSync Dashboard", page_icon="🧠")

with st.sidebar:
    st.markdown("<h2 style='text-align: center; letter-spacing: 2px;'>🧠 MINDSYNC</h2>", unsafe_allow_html=True)
    st.write("---")
    
    page = option_menu(
        menu_title=None, 
        options=["About", "MindSync Engine"], 
        icons=["info-circle", "camera-video"],  
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "0px", 
                "padding": "12px",
                "--hover-color": "rgba(255, 255, 255, 0.05)"
            },
            "nav-link-selected": {"background-color": "#4CAF50", "font-weight": "bold"},
        }
    )
    
    st.write("---")
    st.caption("v1.0 | Edge Inference Active")

if page == "About":
    st.title("ℹ️ The Purpose of MindSync")
    st.write("---")
    
    st.markdown("""
    ### The Core Problem: The Empathy Gap in Software
    Traditional digital learning platforms and productivity tools are highly efficient at delivering content, but they are entirely blind to the user's cognitive state. 
    
    When a student sitting in a classroom becomes confused or frustrated, a human tutor naturally reads their body language and steps in to adjust the pace or explain the concept differently. A standard screen does not. It continues to present information blindly. This "empathy gap" leads to isolation, severe cognitive overload, and high drop-out rates in online education.

    ### The MindSync Solution
    MindSync was built to bridge this gap by giving digital interfaces **emotional intelligence**. 
    
    Rather than waiting for a user to explicitly ask for help, MindSync acts as a proactive, empathetic co-pilot. It uses lightweight, privacy-first edge AI to continuously "read the room." When it detects that a user has hit a wall of friction—such as sustained confusion or distraction—it automatically triggers a generative AI orchestrator.

    ### How It Improves Outcomes
    By observing user states in real-time, MindSync achieves three critical improvements:
    
    * **1. Catching Friction Early:** It identifies confusion exactly when it starts, preventing it from spiraling into frustration or task abandonment.
    * **2. Restoring the Flow State:** When an intervention is needed, the system generates a highly contextual "mental reset"—a pedagogical nudge designed specifically to de-escalate frustration and guide the user back into deep focus.
    * **3. Frictionless Assistance:** The user never has to click a "Help" button. The system adapts to their needs organically, mirroring the experience of working alongside a seasoned human mentor.
    """)
    

elif page == "MindSync Engine":
    
    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.classifier = EmotionClassifier()