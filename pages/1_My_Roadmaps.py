import streamlit as st
from utils.db_functions import get_all_roadmaps
from streamlit_extras.switch_page_button import switch_page

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("assets/style.css")

st.title("My Roadmaps")
# st.markdown("[🏠 Home](/)", unsafe_allow_html=True)

roadmaps = get_all_roadmaps()

if not roadmaps:
    st.info("No roadmaps found. Please generate a roadmap on the Home page.")
else:
    # Build a dictionary mapping display names to roadmap entries.
    roadmap_dict = {f"{entry['topic']}-{entry['num_steps']}-{entry['time_steps']}-{entry['purpose']}": entry for entry in roadmaps}
    selection = st.selectbox("### Select a Roadmap", list(roadmap_dict.keys()))
    
    selected_entry = roadmap_dict[selection]
    st.markdown("##### Table of Contents (Preview):")
    
    # Preview only the titles (Table of Contents)
    roadmap_array = selected_entry.get("roadmap_output", [])
    if roadmap_array:
        toc_html = "<ul>"
        for section in roadmap_array:
            toc_html += f"<li>{section['title']}</li>"
        toc_html += "</ul>"
        toc_html += "<br></br>"
        st.markdown(toc_html, unsafe_allow_html=True)
    else:
        st.info("No detailed roadmap found for this entry.")
    
    if st.button("View Full Roadmap"):
        st.session_state.current_submission = selected_entry
        st.success("Redirecting to the full Roadmap page...")
        switch_page("roadmap")
