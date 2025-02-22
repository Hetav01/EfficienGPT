import streamlit as st
from utils.db_functions import get_db  # To query the database

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
local_css("assets/style.css")

st.title("Roadmap Detail")

# Navigation links
st.markdown("[🏠 Home](/)", unsafe_allow_html=True)
st.markdown("[My Roadmaps](./2_My_Roadmaps)", unsafe_allow_html=True)

# Get query parameters from the URL (e.g. ?id=<roadmap_id>)
query_params = st.query_params
roadmap_id = query_params.get("id", [None])[0]

roadmap = None

if roadmap_id:
    # Fetch the specific roadmap document from MongoDB Atlas
    db = get_db()
    collection = db["roadmaps"]
    roadmap = collection.find_one({"id": roadmap_id})
elif "current_submission" in st.session_state:
    # Fall back to the most recent submission stored in session state
    roadmap = st.session_state.current_submission

if roadmap:
    st.subheader(f"Learning Roadmap for: {roadmap['topic']}")
    st.write(f"**Timeframe:** {roadmap['time_steps']}")
    st.write(f"**Number of Steps:** {roadmap['num_steps']}")
    st.write(f"**Purpose:** {roadmap['purpose']}")
    st.write("### Generated Roadmap")
    st.markdown(roadmap["roadmap_output"], unsafe_allow_html=True)
else:
    st.info("No roadmap selected. Please generate a roadmap on the Home page or select one from 'My Roadmaps'.")
