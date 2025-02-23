import streamlit as st
import time
from utils.llm_response import (
    interview_roadmap_chain, 
    generic_roadmap_roadmap_chain, 
    extract_headings, 
    generate_interview_response, 
    generate_generic_response
)
from utils.db_functions import get_db, insert_roadmap  # For optional DB lookup if needed

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("assets/style.css")

# Updated generate_roadmap function that returns a dictionary
def generate_roadmap(topic, time_steps, num_steps, purpose, role=None, job_description=None):
    
    if purpose == "Interview Prep":
        response = interview_roadmap_chain.invoke({
            "topic_name": topic,
            "input_number": num_steps,
            "time_limit": time_steps,
            "use_case": purpose,
            "role": None,
            "job_description": None
        })
    else:
        response = generic_roadmap_roadmap_chain.invoke({
            "topic_name": topic,
            "input_number": num_steps,
            "time_limit": time_steps,
            "use_case": purpose
        })

    headings = extract_headings(response)

    roadmap_array = []

    for heading in headings:
        ph = st.empty()  # Create a placeholder for the heading status
        # ph.markdown(f"**{heading}**: Generating... :hourglass_flowing_sand:")
        ph.markdown(f'<div class="loading-placeholder"><strong>{heading}</strong>: Generating... :hourglass_flowing_sand:</div>', unsafe_allow_html=True)

        title = heading
        if purpose == "Interview Prep":
            text = generate_interview_response(topic, heading, role, job_description)
        else:
            text = generate_generic_response(topic, heading, purpose)

        roadmap_array.append({
            "title": title,
            "text": text
        })
        ph.markdown(f"**{heading}**: ✅ Completed")

    return roadmap_array



st.title("Roadmap Detail")
# st.markdown("[🏠 Home](/)", unsafe_allow_html=True)
# st.markdown("[My Roadmaps](./My_Roadmaps)", unsafe_allow_html=True)
roadmap_array=[]

if "current_submission" in st.session_state:
    # st.info("No roadmap selected. Please generate one on the Home page or select one from My Roadmaps.")
    submission = st.session_state.current_submission
    topic = submission["topic"]
    time_steps = submission["time_steps"]
    num_steps = submission["num_steps"]
    purpose = submission["purpose"]
    role = submission.get("role", "")
    job_description = submission.get("job_description", "")
    
    st.subheader(f"Learning Roadmap for: {topic}")
    st.write(f"**Timeframe:** {time_steps} | **Number of Steps:** {num_steps} | **Purpose:** {purpose}")
    
    # Display the generated roadmap along with a table of contents
    roadmap_array = submission.get("roadmap_output", [])

elif "submitted" in st.session_state:
    # Save user inputs into session state
    id = st.session_state.submission_id
    topic = st.session_state.topic
    time_steps = st.session_state.time_steps
    num_steps = st.session_state.num_steps
    purpose = st.session_state.purpose
    role = st.session_state.role
    job_description = st.session_state.job_description


    roadmap_array = generate_roadmap(topic, time_steps, num_steps, purpose, role, job_description)

    entry = {
        "id": st.session_state.submission_id,
        "topic": topic,
        "time_steps": time_steps,
        "num_steps": num_steps,
        "purpose": purpose,
        "role": role,
        "job_description": job_description,
        "roadmap_output": roadmap_array,  # now a dictionary of expanded responses
        "timestamp": time.time()
    }

    # Save the submission to MongoDB Atlas using your helper function from db.py
    insert_roadmap(entry)

    if "submitted" in st.session_state:
            del st.session_state["submitted"]

    st.success("Roadmap generation completed!")

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
            st.markdown(f'<a name="{anchor}" style="font-size: 2rem;" >{section["title"]} </a>', unsafe_allow_html=True)
            st.markdown(section["text"], unsafe_allow_html=True)
else:
    st.info("No roadmap content available.")
