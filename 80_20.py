from dotenv import load_dotenv
from warnings import filterwarnings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama

#ignore warnings
filterwarnings("ignore")

# Load environment variables from .env or HOME PATH file
load_dotenv()

# LLama3_2 = ChatOllama(model= "llama3.2:3b-instruct-q2_K")
# LLama3_1 = ChatOllama(model= "llama3.1:8b-instruct-q3_K_L")
openAILLM = ChatOpenAI(model="gpt-4o")

# Define the interview prompt template
interview_80_20_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an expert in condensing complex information into easy to understand concepts and is amazing at explaining those concepts clearly. Just like the Pareto 80/20 Principle. I need to learn about a particular topic in a hurry for an important job interview and you're my ultimate saviour."),
        ("human", 
         """
            The topic I want to learn about is {sub_topic}. The Job interview is for a {job_title} position and the job description is as follows: {job_description}.
            Identify and share the 20%\ of the most important learnings from this sub_topic to help me understand 80%\ of them.
            This subtopic: {sub_topic} is part of learning the main topic: {topic_name} with the purpose of preparing for an interview and should be answered accordingly, so as to not have any overlap with the other subtopics.
            Remember to include any code and resources for the same if necessary in detail. Explain the concepts in using the 80/20 principle.
            If there's no code necessary, just don't include it. For topics that are not technical or don't require the code to be included, just don't include any code.
            The output should only have the learning part and nothing boilerplate starting with: "Here's", etc.
         """)
    ]
)

# Define the interview prompt template
generic_80_20_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an expert in condensing complex information into easy to understand concepts and is amazing at explaining those concepts clearly. Just like the Pareto 80/20 Principle. I need to learn about a particular topic in a hurry for an important job interview and you're my ultimate saviour."),
        ("human", 
         """
            The topic I want to learn about is {sub_topic}. The purpose is {use_case}, and the overarching topic is: {topic_name}.
            Identify and share the 20%\ of the most important learnings from this sub_topic to help me understand 80%\ of them.
            This subtopic: {sub_topic} is part of learning the main topic: {topic_name} with the purpose of {use_case} and should be answered accordingly, so as to not have any overlap with the other subtopics.
            Remember to include any code and resources for the same if necessary in detail. Explain the concepts in using the 80/20 principle.
            If there's no code necessary, just don't include it. For topics that are not technical or don't require the code to be included, just don't include any code.
            The output should only have the learning part and nothing boilerplate starting with: "Here's", etc.
         """)
    ]
)
# Define the list of topics
topics = ['Day 1: Foundations and Basics of BERT', 'Day 2: Generalizing to New Tasks with BERT', 'Day 3: Fine-Tuning BERT for Specific Domains', 'Day 4: Advanced Techniques and Tools for BERT', 'Day 5: Practical Applications and Case Studies with BERT', 'Day 6: Reviewing and Refining BERT for Improved Accuracy']

use_case = "Making a Project"

# Function to generate response for a given topic
def generate_response(topic, template):
    response_chain = template | openAILLM | StrOutputParser()
    
    if use_case.lower() == "interview":
        response = response_chain.invoke({
            "topic_name": "BERT",
            "sub_topic": topic,
            "job_title": "ML Engineer",
            "job_description": """We are looking for a Machine Learning Engineer with expertise in BERT (Bidirectional Encoder Representations from Transformers) and Natural Language Processing (NLP). In this role, you will be responsible for designing, training, fine-tuning, and deploying BERT-based models for various NLP applications such as text classification, entity recognition, sentiment analysis, and question answering. You will work closely with data scientists, software engineers, and product teams to develop scalable and high-performance NLP solutions.

            Key Responsibilities:
            Design and develop NLP models using BERT and its variants (e.g., RoBERTa, DistilBERT, ALBERT).
            Fine-tune pre-trained BERT models on domain-specific datasets to improve accuracy and efficiency.
            Preprocess and clean large-scale text datasets for model training and evaluation.
            Optimize and deploy BERT models in production environments using frameworks like TensorFlow, PyTorch, and Hugging Face Transformers.
            Implement scalable and efficient inference pipelines for real-time and batch processing.
            Conduct experiments, analyze model performance, and iterate on improvements.
            Collaborate with software engineers to integrate NLP models into applications and APIs.
            Stay up-to-date with the latest advancements in NLP and transformer-based models.
            Required Qualifications:
            Bachelor's or Master's degree in Computer Science, AI, Machine Learning, or a related field.
            Strong experience with BERT and transformer-based models.
            Proficiency in Python and machine learning frameworks like TensorFlow, PyTorch, and Hugging Face Transformers.
            Hands-on experience with NLP tasks such as text classification, named entity recognition, and text generation.
            Strong understanding of deep learning, attention mechanisms, and transfer learning.
            Experience deploying ML models using Docker, Kubernetes, and cloud services (AWS, GCP, or Azure).
            Familiarity with ML Ops best practices, including model monitoring and version control.
            Excellent problem-solving skills and the ability to work in a fast-paced, collaborative environment.
            Preferred Qualifications:
            Experience with low-latency model optimization techniques, such as quantization and pruning.
            Familiarity with vector databases (e.g., FAISS, Pinecone) for semantic search applications.
            Background in information retrieval, recommendation systems, or conversational AI.
            Contributions to open-source NLP projects or research publications in NLP conferences (e.g., NeurIPS, ACL, EMNLP).
            Benefits:
            Competitive salary and performance-based bonuses.
            Flexible work hours and remote work options.
            Professional development opportunities and conferences.
            Health, dental, and vision insurance.
            Stock options and employee wellness programs.
            """
    })
        
    else:
        response = response_chain.invoke({
            "topic_name": "BERT",
            "sub_topic": topic,
            "use_case": use_case
        })
        
    return response

# Example usage
for topic in topics:
    response = generate_response(topic, generic_80_20_template)
    print(f"Response for topic '{topic}':\n{response}\n")
    