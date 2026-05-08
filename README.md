🧠 MindSync

Edge-Optimized Affective Computing & Generative Interventions

MindSync is a production-ready, real-time affective computing framework designed to close the feedback loop between human cognitive states and generative AI. It acts as an empathetic digital co-pilot, monitoring user engagement and providing proactive pedagogical nudges when cognitive friction is detected.

🎯 The Purpose: Bridging the "Empathy Gap"

Traditional digital learning platforms and productivity tools are highly efficient at delivering content, but they are entirely blind to the user's cognitive state. When a student sitting in a classroom becomes confused or frustrated, a human tutor naturally reads their body language and steps in to adjust the pace or explain the concept differently. A standard screen does not. It continues to present information blindly.

This "empathy gap" leads to isolation, severe cognitive overload, and high drop-out rates in online education. MindSync was built to bridge this gap by giving digital interfaces emotional intelligence.

💡 The Solution

MindSync does not wait for the user to explicitly click a "Help" button. Instead, it utilizes a sophisticated two-stage pipeline to "read the room" and intervene seamlessly:

Stage 1: The Affective Observer (Edge Inference)

A lightweight computer vision pipeline runs entirely at the edge (in the user's local browser environment). It continuously monitors facial micro-expressions, translating visual data into distinct cognitive states:

🟢 Focused (Flow state optimal)

🟡 Distracted (Attention drifting)

🔴 Confused / Frustrated (High cognitive load)

⚪ Neutral (Baseline)

Privacy-First: Inferences are made locally. No high-definition video frames are ever saved or transmitted to the cloud.

Stage 2: The LLM Orchestrator (Generative Interventions)

A robust state manager buffers the cognitive data to prevent noisy or hyperactive interventions. When a sustained state of friction (e.g., prolonged confusion) is detected, the Orchestrator triggers a Large Language Model (LLM) to synthesize highly contextual "mental resets." These are tailored pedagogical nudges designed specifically to de-escalate frustration and guide the user back into deep focus.

⚙️ Implementation & Architecture

MindSync is built for low-latency, high-performance edge inference on cloud deployment platforms (like Streamlit Community Cloud).

Tech Stack

Frontend & Routing: Streamlit, streamlit-webrtc, streamlit-option-menu

Computer Vision Pipeline: OpenCV (Haar Cascades for face extraction), TensorFlow/Keras

Edge Model: MobileNetV2 (Custom fine-tuned for facial emotion recognition)

Generative AI: Google Gemini 2.5 Flash (via google-generativeai)

Data Persistence (Optional): Supabase

Key Engineering Optimizations

Deploying real-time computer vision to a shared cloud environment requires aggressive optimization. MindSync implements:

Resolution Chokehold: WebRTC media streams are constrained to 320x240 resolution, and audio tracks are disabled, drastically reducing network bandwidth and CPU overhead.

Frame Skipping: The CV pipeline implements selective frame processing (analyzing 1 in every 5 frames) while caching intermediate states, ensuring near 0-latency UI updates without overwhelming the server.

Raw Weight Architecture: To prevent Keras versioning conflicts between the training environment and the deployment server, the MobileNetV2 architecture is explicitly defined in code, bypassing .h5 metadata deserialization bugs.

Asynchronous State Threading: Safely bridges the high-speed WebRTC background processing thread with the main Streamlit UI thread via buffered st.session_state synchronization and st.empty() placeholders.

🚀 Getting Started (Local Development)

1. Clone the Repository

git clone [https://github.com/Aalaukik/mindsync.git](https://github.com/Aalaukik/mindsync.git)
cd mindsync


2. Set Up the Environment

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

3. Configure Secrets

Create a .env file in the root directory and add your Gemini API Key:

GEMINI_API_KEY=your_actual_api_key_here

4. Run the Application

streamlit run app.py

Developed as a high-performance framework for empathetic human-computer interaction.
