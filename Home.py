import streamlit as st
import uuid
from dotenv import load_dotenv
from utils.db_functions import insert_roadmap  # For storing the roadmap if needed
from streamlit_extras.switch_page_button import switch_page

load_dotenv()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("assets/style.css")

st.title("EfficienGPT")

st.write("### What is it that you want to conquer?")
topic = st.text_input("Enter your topic name:")

st.write("### In what time?")
time_steps = st.selectbox("Select time frame:", ("Hours", "Days", "Weeks"))

st.write("### Number of Steps (1-24)")
num_steps = st.number_input("Enter number of steps:", min_value=1, max_value=24, value=3, step=1)

st.write("### What is your purpose?")
purpose = st.selectbox("Select your purpose:", ("Interview Prep", "Project", "Upskill"))

# Add a switch for model selection.
st.write("### Select LLM Model")
model_choice = st.radio("Choose your model:", ["ChatGPT", "Ollama"], index=0, horizontal=True, help="Default is ChatGPT (gpt-4o)")
# Save the chosen model into session state
st.session_state.selected_model = model_choice

role = ""
job_description = ""
if purpose == "Interview Prep":
    st.write("#### Additional Details for Interview Prep (Optional)")
    role = st.text_input("Enter the target role:")
    job_description = st.text_area("Enter the job description:", height=150)

if st.button("Submit"):
    if topic.strip() == "":
        st.error("Please enter a topic name.")
    else:
        # Save user inputs into session state along with a unique submission id.
        st.session_state.submission_id = str(uuid.uuid4())
        st.session_state.topic = topic
        st.session_state.time_steps = time_steps
        st.session_state.num_steps = num_steps
        st.session_state.purpose = purpose
        st.session_state.role = role
        st.session_state.job_description = job_description
        # Clear any previous generated roadmap output
        if "current_submission" in st.session_state:
            del st.session_state["current_submission"]
        if "roadmap_output" in st.session_state:
            del st.session_state["roadmap_output"]
        st.session_state.submitted = True
        st.success("Your details have been saved!")
        st.info("Redirecting you to the Roadmap page...")
        switch_page("roadmap")
