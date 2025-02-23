import streamlit as st
from utils.db_functions import get_db

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

st.title("Roadmap Detail")
st.markdown("[🏠 Home](/)", unsafe_allow_html=True)
st.markdown("[My Roadmaps](./1_My_Roadmaps)", unsafe_allow_html=True)

# Retrieve the roadmap: first by query parameter, then falling back to session state.
query_params = st.query_params
roadmap_id = query_params.get("id", [None])[0]

roadmap = None
if roadmap_id:
    db = get_db()
    collection = db["roadmaps"]
    roadmap = collection.find_one({"id": roadmap_id})
elif "current_submission" in st.session_state:
    roadmap = st.session_state.current_submission

if roadmap:
    st.subheader(f"Learning Roadmap for: {roadmap['topic']}")
    st.write(f"**Timeframe:** {roadmap['time_steps']}")
    st.write(f"**Number of Steps:** {roadmap['num_steps']}")
    st.write(f"**Purpose:** {roadmap['purpose']}")
    
    roadmap_array = roadmap.get("roadmap_output", [])
    if roadmap_array:
        # Create a Table of Contents (TOC)
        st.markdown("## Table of Contents")
        toc_html = "<ul>"
        for section in roadmap_array:
            # Skip the header section in the TOC if desired; here we assume it's the first element.
            if section["title"].lower().startswith("header"):
                continue
            # Create an anchor ID by replacing spaces with hyphens
            anchor = section["title"].replace(" ", "-")
            toc_html += f'<li><a href="#{anchor}">{section["title"]}</a></li>'
        toc_html += "</ul>"
        st.markdown(toc_html, unsafe_allow_html=True)
        
        # Now iterate through and display each section
        for section in roadmap_array:
            # For the header section, simply display its text without an anchor
            if section["title"].lower().startswith("header"):
                st.markdown(section["text"], unsafe_allow_html=True)
            else:
                # Insert an HTML anchor for internal navigation
                anchor = section["title"].replace(" ", "-")
                st.markdown(f'<a name="{anchor}"></a>', unsafe_allow_html=True)
                st.header(section["title"])
                st.markdown(section["text"], unsafe_allow_html=True)
    else:
        st.info("No roadmap content available.")
else:
    st.info("No roadmap selected. Please generate a roadmap or select one from 'My Roadmaps'.")
