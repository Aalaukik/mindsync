import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import time
import pandas as pd
from streamlit_option_menu import option_menu
from src.cv_pipeline import EmotionClassifier
from src.state_manager import CognitiveBuffer
from src.llm_orchestrator import MindSyncOrchestrator

if "buffer" not in st.session_state:
    st.session_state.buffer = CognitiveBuffer()
else:
    if not hasattr(st.session_state.buffer, 'friction_start_time'):
        st.session_state.buffer = CognitiveBuffer()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MindSyncOrchestrator()
if "current_nudge" not in st.session_state:
    st.session_state.current_nudge = ""

if "focus_start_time" not in st.session_state:
    st.session_state.focus_start_time = None
if "unfocused_start_time" not in st.session_state:
    st.session_state.unfocused_start_time = None
if "max_focus_time" not in st.session_state:
    st.session_state.max_focus_time = 0
if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = time.time()
if "interventions_triggered" not in st.session_state:
    st.session_state.interventions_triggered = 0

st.set_page_config(layout="wide", page_title="MindSync Dashboard", page_icon="🧠")

with st.sidebar:
    st.markdown("<h2 style='text-align: center; letter-spacing: 2px;'>🧠 MINDSYNC</h2>", unsafe_allow_html=True)
    st.write("---")
    
    page = option_menu(
        menu_title=None, 
        options=["About MindSync", "MindSync Engine", "Session Analytics"], 
        icons=["info-circle", "camera-video", "graph-up"],  
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "padding": "12px"},
            "nav-link-selected": {"background-color": "#4CAF50", "font-weight": "bold"},
        }
    )
    st.write("---")
    st.caption("v1.3 | HD Frontend / Edge Backend")

if page == "About MindSync":
    st.title("🧠 MindSync: Bridging the Empathy Gap in AI")
    st.write("---")
    
    st.markdown("""
    ### The Core Problem: Digital Isolation
    Traditional digital learning platforms and productivity tools are highly efficient at delivering content, but they are entirely **blind to the user's cognitive state**. 
    
    When a student sitting in a physical classroom becomes confused or frustrated, a human tutor naturally reads their facial expressions and steps in to adjust the pace, offer encouragement, or explain the concept differently. A standard screen does not. It continues to present information blindly. This "empathy gap" leads to severe cognitive overload, feelings of isolation, and high drop-out rates in online education and remote work.

    ### The MindSync Solution
    MindSync was engineered to solve this by giving digital interfaces **emotional intelligence**. 
    
    Rather than waiting for a user to explicitly click a "Help" button, MindSync acts as a proactive, empathetic AI co-pilot. It uses a dual-pipeline architecture to continuously "read the room" and intervene exactly when needed.

    ### How The Architecture Works
    1. **Edge Computer Vision (The Observer):** MindSync uses a lightweight, privacy-first affective computing model running entirely on the edge. It tracks macro-expressions to determine if the user is in a state of Flow (Focused/Neutral) or Friction (Confused/Frustrated). 
    2. **The Pure Stopwatch Buffer:** To prevent UI flicker and hardware lag from ruining the experience, an absolute-time cognitive buffer analyzes the data. If sustained friction is detected for exactly 1.0 seconds, the system locks.
    3. **Generative LLM Orchestrator (The Tutor):** Once triggered, a highly optimized Gemini API call synthesizes a real-time "mental reset." This pedagogical nudge is designed specifically to de-escalate frustration and guide the user back into deep focus.

    ### Key Features
    * **Frictionless Assistance:** The system adapts to user needs organically, mirroring the experience of working alongside a seasoned human mentor.
    * **Flow State Gamification:** Users are incentivized to maintain deep work through a real-time, time-based focus streak multiplier.
    * **Post-Session Analytics:** Every session is tracked and visualized in a dedicated data dashboard, allowing users to review their cognitive load timelines and pinpoint exact moments of distraction.
    """)

elif page == "MindSync Engine":
    
    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.classifier = EmotionClassifier()
            self.latest_state = "Neutral" 
                       
            self.frame_skip = 5 
            self.frame_count = 0
            
        def recv(self, frame):           
            img = frame.to_ndarray(format="bgr24")
            
            if self.frame_count % self.frame_skip == 0:
                h, w, _ = img.shape                
               
                start_x = (w - h) // 2
                square_img = img[:, start_x : start_x + h] 
                                
                ml_ready_img = cv2.resize(square_img, (224, 224))           
                                
                state = self.classifier.predict_frame(ml_ready_img)
                if state:
                    self.latest_state = state
                    
            self.frame_count += 1
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    st.title("🧠 MindSync Engine")
    st.write("---")

    col_video, col_analytics = st.columns([2, 1], gap="large")

    with col_video:        
        ctx = webrtc_streamer(
            key="mindsync-eye",
            video_processor_factory=EmotionProcessor,
            async_processing=True,
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, 
                "audio": False
            },
            video_html_attrs={
                "style": {"width": "100%", "transform": "scaleX(-1)", "border-radius": "10px"}, # Mirrors the feed!
                "controls": False,
                "autoPlay": True,
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    with col_analytics:
        st.markdown("### 📊 Cognitive State")
        emotion_placeholder = st.empty()
        
        st.markdown("### 🔥 Flow Streak")
        streak_placeholder = st.empty()
        
        st.markdown("### 💡 Orchestrator Output")
        nudge_placeholder = st.empty()
        nudge_placeholder.info("Awaiting camera stream...")

    if ctx and ctx.state.playing:
        while True:
            if ctx.video_processor:
                current_state = ctx.video_processor.latest_state
                current_time = time.time()
                
                clean_state = str(current_state).strip().title()
                flow_states = ["Focused", "Neutral"]
                
                with emotion_placeholder.container():
                    if clean_state in ["Confused", "Frustrated"]:
                        st.error(f"## {clean_state} 📉")
                    elif clean_state == "Focused":
                        st.success(f"## {clean_state} 🎯")
                    elif clean_state == "Distracted":
                        st.warning(f"## {clean_state} 👀")
                    else:
                        st.info(f"## {clean_state} 😐")
                
                if clean_state in flow_states:
                    if st.session_state.focus_start_time is None:
                        st.session_state.focus_start_time = current_time
                    st.session_state.unfocused_start_time = None  
                    
                    current_streak = int(current_time - st.session_state.focus_start_time)
                    if current_streak > st.session_state.max_focus_time:
                        st.session_state.max_focus_time = current_streak
                else:
                    if st.session_state.unfocused_start_time is None:
                        st.session_state.unfocused_start_time = current_time
                        
                    if (current_time - st.session_state.unfocused_start_time) > 1.5:
                        st.session_state.focus_start_time = None
                        current_streak = 0
                    else:
                        if st.session_state.focus_start_time is not None:
                            current_streak = int(current_time - st.session_state.focus_start_time)
                        else:
                            current_streak = 0

                with streak_placeholder.container():
                    st.metric("Consecutive Focus", f"{current_streak} sec", f"High Score: {st.session_state.max_focus_time} sec")

                if len(st.session_state.session_log) == 0 or (current_time - st.session_state.session_log[-1]["timestamp"] >= 1.0):
                    st.session_state.session_log.append({
                        "timestamp": current_time,
                        "time_elapsed": round(current_time - st.session_state.session_start_time, 1),
                        "state": clean_state 
                    })
                
                st.session_state.buffer.add_state(clean_state)
                needs_help, emotion = st.session_state.buffer.requires_intervention()
                
                if needs_help:
                    st.session_state.interventions_triggered += 1
                    with nudge_placeholder.container():
                        with st.spinner("Synthesizing mental reset..."):
                            nudge = st.session_state.orchestrator.generate_nudge(
                                emotion=emotion, topic="Independent Work", current_content="User is engaged in a task."
                            )
                            st.session_state.current_nudge = nudge
                
                if current_streak > 30:
                    st.session_state.current_nudge = ""

                if st.session_state.current_nudge:
                    nudge_placeholder.success(f"**Agent:** {st.session_state.current_nudge}")
                else:
                    nudge_placeholder.info("No friction detected. Flow state optimal.")

            time.sleep(0.05)

elif page == "Session Analytics":
    st.title("📈 Post-Session Analytics")
    st.write("Review your cognitive load and flow state performance.")
    st.write("---")
    
    if len(st.session_state.session_log) < 5:
        st.warning("Not enough data. Start the MindSync Engine and record a session for at least a few seconds!")
    else:
        df = pd.DataFrame(st.session_state.session_log)
        
        emotion_weights = {"Focused": 3, "Neutral": 2, "Distracted": 1, "Confused": 0, "Frustrated": 0}
        df["Cognitive Score"] = df["state"].map(emotion_weights)
        
        total_time = df["time_elapsed"].max()
        focus_time = len(df[df["state"].isin(["Focused", "Neutral"])]) 
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Session Time", f"{total_time} sec")
        col2.metric("Total Focus Time", f"{focus_time} sec")
        col3.metric("Interventions Triggered", f"{st.session_state.interventions_triggered}")
        
        st.write("---")
        st.markdown("### 🧠 Cognitive Load Timeline")
        st.markdown("*3 = Focused | 2 = Neutral | 1 = Distracted | 0 = Friction*")
        
        chart_data = df.set_index("time_elapsed")[["Cognitive Score"]]
        st.area_chart(chart_data, color="#4CAF50")
        
        if st.button("Reset Session Data"):
            st.session_state.session_log = []
            st.session_state.session_start_time = time.time()
            st.session_state.focus_start_time = None
            st.session_state.unfocused_start_time = None
            st.session_state.max_focus_time = 0
            st.session_state.interventions_triggered = 0
            st.rerun()