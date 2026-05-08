import streamlit as st

def render_header():
    """Renders the main application header and privacy notice."""
    st.title("🧠 MindSync: Affective Virtual Tutoring")
    st.markdown("""
    **Privacy First:** Video is processed locally on the edge. 
    Only metadata is utilized for pedagogical intervention. No images are stored.
    """)
    st.write("---")

def render_mindsync_sidebar(current_nudge: str, current_state: str):
    """
    Renders the non-intrusive sidebar for MindSync interventions.
    """
    st.sidebar.title("MindSync Mentor")    
   
    if current_state:
        st.sidebar.markdown(f"**Current State:** `{current_state}`")
    else:
        st.sidebar.markdown("**Current State:** `Awaiting Camera...`")
        
    st.sidebar.divider()    
    
    st.sidebar.subheader("Active Nudge")
    if current_nudge:
        st.sidebar.info(current_nudge)
    else:
        st.sidebar.write("Flow state optimal. Keep going!")

def render_impact_metrics(cr_control: float = 45.0, cr_experimental: float = 60.0):
    """
    Renders the A/B testing metrics and Persistence Lift calculation.
    Uses default mock data for demonstration.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Strategic Impact Metrics")
       
    if cr_control > 0:
        persistence_lift = ((cr_experimental - cr_control) / cr_control) * 100
    else:
        persistence_lift = 0.0

    st.sidebar.latex(r"Persistence Lift = \frac{CR_{exp} - CR_{ctrl}}{CR_{ctrl}} \times 100")
        
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Control CR", f"{cr_control}%")
    col2.metric("Experimental CR", f"{cr_experimental}%")
    
    st.sidebar.metric(
        label="Persistence Lift", 
        value=f"{persistence_lift:.1f}%", 
        delta=f"+{persistence_lift:.1f}% improvement"
    )