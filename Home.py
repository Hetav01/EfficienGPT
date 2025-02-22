import streamlit as st
import time
import uuid
from utils.db_functions import insert_roadmap  # Import the insert function
from utils.llm_response import roadmap_chain

# Function to load custom CSS
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
num_steps = st.number_input("Enter number of steps:", min_value=1, max_value=24, value=1, step=1)

st.write("### What is the need for learning? (Interview Prep, Project, Upskill)")
purpose = st.selectbox("Select your purpose:", ("Interview Prep", "Project", "Upskill"))

if st.button("Submit"):
    if topic.strip() == "":
        st.error("Please enter a topic name.")
    else:
        with st.spinner("Generating roadmap..."):
            st.session_state.roadmap_output = roadmap_chain.invoke({
                "topic_name": topic,
                "input_number": num_steps,
                "time_limit": time_steps,
                "use_case": purpose
            })
        submission_id = str(uuid.uuid4())
        entry = {
            "id": submission_id,
            "topic": topic,
            "time_steps": time_steps,
            "num_steps": num_steps,
            "purpose": purpose,
            "roadmap_output": st.session_state.roadmap_output,
            "timestamp": time.time()
        }
        # Save the submission using the MongoDB function from db.py
        insert_roadmap(entry)
        
        st.success("Your details have been saved and a roadmap has been generated!")
        st.info("You can view all your roadmaps on the 'My Roadmaps' page from the sidebar.")
