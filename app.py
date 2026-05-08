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
            self.latest_state = "Neutral" 
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            state = self.classifier.predict_frame(img)
            
            if state:
                self.latest_state = state
                
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    st.title("🧠 MindSync Engine")
    st.markdown("Real-time affective computing and generative interventions.")
    st.write("---")

    col_video, col_analytics = st.columns([2, 1], gap="large")

    with col_video:
        st.markdown("### 🎥 Edge Inference Feed")
        
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

    with col_analytics:
        st.markdown("### 📊 Cognitive State")
        emotion_placeholder = st.empty()
        emotion_placeholder.info("Awaiting camera stream...")
        
        st.markdown("### 💡 Orchestrator Output")
        nudge_placeholder = st.empty()
        nudge_placeholder.info("No friction detected.")

    if ctx and ctx.state.playing:
        while True:
            if ctx.video_processor:
                current_state = ctx.video_processor.latest_state
                
                with emotion_placeholder.container():
                    if current_state in ["Confused", "Frustrated"]:
                        st.error(f"## {current_state} 📉\n**Status:** High Cognitive Load")
                    elif current_state == "Focused":
                        st.success(f"## {current_state} 🎯\n**Status:** Flow State Optimal")
                    elif current_state == "Distracted":
                        st.warning(f"## {current_state} 👀\n**Status:** Attention Drifting")
                    else:
                        st.info(f"## {current_state} 😐\n**Status:** Baseline")
                
                st.session_state.buffer.add_state(current_state)
                
                needs_help, emotion = st.session_state.buffer.requires_intervention()
                
                if needs_help:
                    with nudge_placeholder.container():
                        with st.spinner("Orchestrator synthesizing mental reset..."):
                            nudge = st.session_state.orchestrator.generate_nudge(
                                emotion=emotion, 
                                topic="Independent Work", 
                                current_content="User is engaged in an active task."
                            )
                            st.session_state.current_nudge = nudge
                
                # Automatically clear the nudge if the user is focused again
                if current_state == "Focused":
                    st.session_state.current_nudge = ""

                if st.session_state.current_nudge:
                    nudge_placeholder.success(f"**Agent:** {st.session_state.current_nudge}")
                else:
                    nudge_placeholder.info("No friction detected. Flow state optimal.")

            time.sleep(0.05)