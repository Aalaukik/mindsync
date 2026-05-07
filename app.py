import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import time
from src.cv_pipeline import EmotionClassifier
from src.state_manager import CognitiveBuffer
from src.llm_orchestrator import MindSyncOrchestrator
from components.ui_elements import render_header, render_impact_metrics

# 1. Initialize session states safely
if "buffer" not in st.session_state:
    st.session_state.buffer = CognitiveBuffer()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MindSyncOrchestrator()
if "current_nudge" not in st.session_state:
    st.session_state.current_nudge = ""

# Page Config for a wider, app-like feel
st.set_page_config(layout="wide", page_title="MindSync Tutor", page_icon="🧠")

# Mock module data
CURRENT_TOPIC = "Data Structures: Hash Tables"
CURRENT_TEXT = "A Hash Table uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. It allows for highly efficient data retrieval."

# 2. Updated Video Processor (Clean Feed, No Green Text)
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.classifier = EmotionClassifier()
        self.latest_state = "Neutral" 
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        state = self.classifier.predict_frame(img)
        
        if state:
            self.latest_state = state
            
        # We removed cv2.putText here to keep the video feed clean and professional!
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Render Header
render_header()

# 4. Modern Dashboard Layout
col_video, col_content = st.columns([1, 2], gap="large")

with col_video:
    st.markdown("### 🎥 Affective Observer")
    
    # Initialize WebRTC with extreme performance constraints
    ctx = webrtc_streamer(
        key="mindsync-eye",
        video_processor_factory=EmotionProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},
            "audio": False 
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
    
    # Dedicated placeholder for our external emotion metric
    st.markdown("### 📊 Live Analytics")
    emotion_placeholder = st.empty()

with col_content:
    # Learning Material Card
    with st.container(border=True):
        st.markdown(f"## 📚 {CURRENT_TOPIC}")
        st.write(CURRENT_TEXT)
    
    st.markdown("### 🧠 MindSync Mentor")
    # Dedicated placeholder for the LLM interventions
    nudge_placeholder = st.empty()
    nudge_placeholder.info("✨ Flow state optimal. Keep going!")

# Render bottom metrics (so they appear before the loop locks the thread)
render_impact_metrics()

# 5. Real-Time UI Synchronization Loop
# This loop actively pulls the emotion from the video thread and updates the UI instantly
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            current_state = ctx.video_processor.latest_state
            
            # Update the separate Emotion Variable cleanly
            with emotion_placeholder.container():
                if current_state in ["Confused", "Frustrated"]:
                    st.error(f"**Cognitive State:** {current_state} 📉")
                elif current_state == "Focused":
                    st.success(f"**Cognitive State:** {current_state} 🎯")
                elif current_state == "Distracted":
                    st.warning(f"**Cognitive State:** {current_state} 👀")
                else:
                    st.info(f"**Cognitive State:** {current_state} 😐")
            
            # Log to buffer
            st.session_state.buffer.add_state(current_state)
            
            # Check for LLM trigger
            needs_help, emotion = st.session_state.buffer.requires_intervention()
            
            if needs_help:
                with nudge_placeholder.container():
                    with st.spinner("Analyzing friction and generating mental reset..."):
                        nudge = st.session_state.orchestrator.generate_nudge(
                            emotion=emotion, 
                            topic=CURRENT_TOPIC, 
                            current_content=CURRENT_TEXT
                        )
                        st.session_state.current_nudge = nudge
            
            # Display active nudge if one exists
            if st.session_state.current_nudge:
                nudge_placeholder.success(f"**Intervention:** {st.session_state.current_nudge}")

        # Sleep briefly to prevent the while-loop from maxing out the CPU
        time.sleep(0.5)