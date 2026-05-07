import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
from src.cv_pipeline import EmotionClassifier
from src.state_manager import CognitiveBuffer
from src.llm_orchestrator import MindSyncOrchestrator
from components.ui_elements import render_header, render_mindsync_sidebar, render_impact_metrics

# 1. Initialize session states safely
if "buffer" not in st.session_state:
    st.session_state.buffer = CognitiveBuffer()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MindSyncOrchestrator()
if "current_nudge" not in st.session_state:
    st.session_state.current_nudge = ""
if "current_state" not in st.session_state:
    st.session_state.current_state = "Awaiting Camera..."

st.set_page_config(layout="wide", page_title="MindSync Learning")

# Mock module data for context scraping
CURRENT_TOPIC = "Data Structures"
CURRENT_TEXT = "A Hash Table uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found."

# 2. Updated Video Processor (Fixes thread crash and deprecation warnings)
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.classifier = EmotionClassifier()
        self.latest_state = "Neutral" # Store state locally in the thread
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        state = self.classifier.predict_frame(img)
        
        if state:
            self.latest_state = state
            
        # UI Transparency: Show tracking is active on the video feed
        cv2.putText(img, f"MindSync Active - State: {self.latest_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Must return an 'av' VideoFrame in the new API
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Render Header
render_header()

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Current Module: {CURRENT_TOPIC}")
    st.write(CURRENT_TEXT)
    st.write("---")
    
    # 4. Initialize WebRTC with new API arguments
    ctx = webrtc_streamer(
        key="mindsync-eye",
        video_processor_factory=EmotionProcessor,
        async_processing=True
    )

# 5. Safe Thread Syncing
# If the video is running, safely extract the state from the processor into the main thread
if ctx.video_processor:
    detected_state = ctx.video_processor.latest_state
    st.session_state.current_state = detected_state
    st.session_state.buffer.add_state(detected_state)

with col2:
    # Check if intervention is needed based on the rolling buffer
    needs_help, emotion = st.session_state.buffer.requires_intervention()
    
    if needs_help:
        with st.spinner("Generating mental reset..."):
            nudge = st.session_state.orchestrator.generate_nudge(
                emotion=emotion, 
                topic=CURRENT_TOPIC, 
                current_content=CURRENT_TEXT
            )
            st.session_state.current_nudge = nudge
            
    # Render Sidebar via components/ui_elements.py
    render_mindsync_sidebar(st.session_state.current_nudge, st.session_state.current_state)

# Render bottom metrics
render_impact_metrics()