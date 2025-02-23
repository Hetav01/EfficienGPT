import streamlit as st
from utils.db_functions import get_all_roadmaps

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

st.title("My Roadmaps")
st.markdown("[🏠 Home](/)", unsafe_allow_html=True)

# Retrieve all stored roadmap entries from MongoDB Atlas
roadmaps = get_all_roadmaps()

if not roadmaps:
    st.info("No roadmaps found. Please generate a roadmap on the Home page.")
else:
    # Build a dictionary mapping display names to roadmap entries
    roadmap_dict = {f"{entry['topic']}-{entry['num_steps']}-{entry['time_steps']}-{entry['purpose']}": entry for entry in roadmaps}
    # Only display a selectbox for the user to choose a roadmap
    selection = st.selectbox("Select a Roadmap", list(roadmap_dict.keys()))
    
    # When the user clicks the button, store the selected entry and update query parameters
    if st.button("View Roadmap Detail"):
        selected_entry = roadmap_dict[selection]
        st.session_state.current_submission = selected_entry
        st.query_params[id]=selected_entry["id"]
        st.success("Roadmap detail updated! Please navigate to the Roadmap Detail page to view full details.")
