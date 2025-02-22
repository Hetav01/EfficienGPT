import streamlit as st
from utils.db_functions import get_all_roadmaps  # Import the retrieval function

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

st.title("My Roadmaps")
st.markdown("[🏠 Home](/)", unsafe_allow_html=True)

roadmaps = get_all_roadmaps()

if not roadmaps:
    st.info("No roadmaps found. Please generate a roadmap on the Home page.")
else:
    # Create a dictionary mapping display names to roadmap entries
    roadmap_dict = {f"Roadmap - {entry['topic']}": entry for entry in roadmaps}
    selection = st.selectbox("Select a Roadmap", list(roadmap_dict.keys()))
    
    if selection:
        selected_entry = roadmap_dict[selection]
        st.subheader(f"Learning Roadmap for: {selected_entry['topic']}")
        st.write(f"**Timeframe:** {selected_entry['time_steps']}")
        st.write(f"**Number of Steps:** {selected_entry['num_steps']}")
        st.write(f"**Purpose:** {selected_entry['purpose']}")
        st.write("### Generated Roadmap")
        st.markdown(selected_entry["roadmap_output"], unsafe_allow_html=True)
