import streamlit as st
import time
import uuid
from utils.db_functions import insert_roadmap   # Import the MongoDB insert function
from utils.llm_response import interview_roadmap_chain, generic_roadmap_roadmap_chain, extract_headings, generate_interview_response, generate_generic_response
from dotenv import load_dotenv

load_dotenv()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

st.title("Title Name")

st.write("### What do you want to learn?")
topic = st.text_input("Enter your topic name:")

st.write("### In how many steps? (Hours, Days, Weeks)")
time_steps = st.radio("Select time frame:", ("Hours", "Days", "Weeks"))

st.write("### Number of Steps (1-24)")
num_steps = st.number_input("Enter number of steps:", min_value=1, max_value=24, value=6, step=1)

st.write("### What is the need for learning? (Interview Prep, Project, Upskill)")
purpose = st.selectbox("Select your purpose:", ("Interview Prep", "Project", "Upskill"))

role = ""
job_description = ""
if purpose == "Interview Prep":
    st.write("#### Additional Details for Interview Prep (Optional)")
    role = st.text_input("Enter the target role:")
    job_description = st.text_area("Enter the job description:", height=150)

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
        title = heading
        if purpose == "Interview Prep":
            text = generate_interview_response(topic, heading, role, job_description)
        else:
            text = generate_generic_response(topic, heading, purpose)

        roadmap_array.append({
            "title": title,
            "text": text
        })

    return roadmap_array
        
if st.button("Submit"):
    if topic.strip() == "":
        st.error("Please enter a topic name.")
    else:
        with st.spinner("Generating roadmap..."):
            roadmap_output = generate_roadmap(topic, time_steps, num_steps, purpose, role, job_description)
        submission_id = str(uuid.uuid4())
        entry = {
            "id": submission_id,
            "topic": topic,
            "time_steps": time_steps,
            "num_steps": num_steps,
            "purpose": purpose,
            "role": role,
            "job_description": job_description,
            "roadmap_output": roadmap_output,  # now a dictionary of expanded responses
            "timestamp": time.time()
        }
        # Save the submission to MongoDB Atlas using your helper function from db.py
        insert_roadmap(entry)
        
        # Optionally store the submission in session state for immediate use
        st.session_state.current_submission = entry
        
        st.success("Your details have been saved and a roadmap has been generated!")
        st.info("You can view your roadmap on the Roadmap Detail page or in 'My Roadmaps'.")
