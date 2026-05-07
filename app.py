import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import time
from src.cv_pipeline import EmotionClassifier
from src.state_manager import CognitiveBuffer
from src.llm_orchestrator import MindSyncOrchestrator

# 1. Initialize session states safely
if "buffer" not in st.session_state:
    st.session_state.buffer = CognitiveBuffer()
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MindSyncOrchestrator()
if "current_nudge" not in st.session_state:
    st.session_state.current_nudge = ""

# Page Config for a wider, immersive dashboard feel
st.set_page_config(layout="wide", page_title="MindSync Dashboard", page_icon="🧠")

# --- NAVIGATION ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select Page:", ["MindSync Engine", "About"])
st.sidebar.write("---")
st.sidebar.caption("MindSync v1.0 | Edge Inference Active")


# ==========================================
# PAGE 1: MINDSYNC ENGINE
# ==========================================
if page == "MindSync Engine":
    
    # 2. Updated Video Processor
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

    # 3. Clean Dashboard Header
    st.title("🧠 MindSync Engine")
    st.markdown("Real-time affective computing and generative interventions.")
    st.write("---")

    # 4. Pro-Layout (Video gets 2/3 of the screen, Analytics gets 1/3)
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
        # Clean, separate placeholder for the emotion state
        emotion_placeholder = st.empty()
        emotion_placeholder.info("Awaiting camera stream...")
        
        st.markdown("### 💡 Orchestrator Output")
        nudge_placeholder = st.empty()
        nudge_placeholder.info("No friction detected.")

    # 5. High-Speed UI Synchronization Loop
    if ctx and ctx.state.playing:
        while True:
            if ctx.video_processor:
                current_state = ctx.video_processor.latest_state
                
                # Snappy, stylized metric updates
                with emotion_placeholder.container():
                    if current_state in ["Confused", "Frustrated"]:
                        st.error(f"## {current_state} 📉\n**Status:** High Cognitive Load")
                    elif current_state == "Focused":
                        st.success(f"## {current_state} 🎯\n**Status:** Flow State Optimal")
                    elif current_state == "Distracted":
                        st.warning(f"## {current_state} 👀\n**Status:** Attention Drifting")
                    else:
                        st.info(f"## {current_state} 😐\n**Status:** Baseline")
                
                # Log to buffer
                st.session_state.buffer.add_state(current_state)
                
                # Check for LLM trigger
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
                
                # Display active nudge
                if st.session_state.current_nudge:
                    nudge_placeholder.success(f"**Agent:** {st.session_state.current_nudge}")

            # 50ms delay for near-instant UI syncing
            time.sleep(0.05) 


# ==========================================
# PAGE 2: ABOUT
# ==========================================
elif page == "About":
    st.title("ℹ️ About MindSync")
    st.write("---")
    
    st.markdown("""
    ### System Architecture
    MindSync is a production-oriented affective computing framework designed to close the feedback loop between human cognitive states and generative AI. 
    
    Rather than relying on explicit user prompts, this system utilizes a two-stage pipeline:
    
    1. **Stage 1: The Affective Observer (Edge Inference)**
    A lightweight computer vision pipeline (MobileNetV2) runs at the edge. It continuously monitors facial micro-expressions, translating visual data into distinct cognitive states (Focused, Distracted, Confused, Frustrated, Neutral). To optimize for latency and privacy, inferences are made without saving or transmitting high-definition video frames to the cloud.
    
    2. **Stage 2: The LLM Orchestrator (Generative Interventions)**
    A state manager buffers the cognitive data to prevent noisy interventions. When a sustained state of friction is detected, the Orchestrator triggers an LLM to synthesize highly contextual nudges designed to seamlessly guide the user back into an optimal flow state.
    
    ### Future Scope
    * **Sensor Fusion:** Exploring the integration of spatial data or keystroke dynamics to enhance the confidence score of the affective observer.
    * **Agentic Orchestration:** Allowing the LLM to trigger active software state changes rather than just passive text nudges.
    """)