from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama
import os

load_dotenv()

# model = ChatOllama(model= "llama3.2:3b-instruct-q2_K")
# model = ChatOllama(model= "llama3.1:8b-instruct-q3_K_L")
model = ChatOpenAI(model="gpt-4o")

generic_roadmap_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in creating learning roadmaps for various topics. Your task is to condense the learning of {topic_name} into {input_number} {time_limit}. Include all relevant subtopics and sub-sub-topics."),
        ("human", """Create a comprehensive roadmap to learn {topic_name} in {input_number} {time_limit}. This is for {use_case} with a deadline of {input_number} {time_limit}.
                Provide detailed headings for each subtopic of {topic_name}. Provide a realistic time it might take to learn these subtopics in the headings too. 
                Ensure the workload is realistically manageable within {input_number} {time_limit}.
                The output should be the roadmap only with just the headings, nothing else.. Don't add anything else to the headings. 
                
                """
                """
                For every different time_limit, the output should a bit difference
                A sample output template example with the topic as Machine Learning, number of days it took as 4 and the use case as Up Skilling is:                  
                    ### Day 1: Foundations and Basics of Machine Learning 

                    ### Day 2: Supervised Learning 

                    ### Day 3: Unsupervised Learning and Model Evaluation 
                    
                    ### Day 4: Advanced Topics and Tools 

                """
                
                """
                   For every different time_limit, the output should a bit difference
                A sample output template example with the topic as Machine Learning, number of weeks it took as 4 and the use case as making a project is:                  
                    ### Week 1: Foundations and Basics of Machine Learning 

                    ### Week 2: Supervised Learning 

                    ### Week 3: Unsupervised Learning and Model Evaluation 
                    
                    ### Week 4: Advanced Topics and Tools 

                """
                
                """
                    If the input is 2 hours, the output should be like this:
                    ## Hour 1: Foundations and Basics of Machine Learning
                    
                    ## Hour 2: Supervised Learning
                    
                    Nothing more.
                """
                
                """
                    If the input is 1 day/1 week, the output should have the headings as hours/days respectively.
                    
                    For 1 day:
                    ## Hour 1: Foundations and Basics of Machine Learning
                    
                    ## Hour 2: Supervised Learning
                    
                    ## Hour 3: Unsupervised Learning and Model Evaluation
                   
                    ## Hour 4: Advanced Topics and Tools
                    
                    ## Hour 5: Practical Applications and Interview Preparation
                    
                    ## Hour 6: Interview Preparation and Mock Interviews
                    
                    For 1 week:
                    ## Day 1: Foundations and Basics of Machine Learning
                    
                    ## Day 2: Supervised Learning
                    
                    ## Day 3: Unsupervised Learning and Model Evaluation
                    
                    ## Day 4: Advanced Topics and Tools
                    
                    ## Day 5: Practical Applications and Interview Preparation
                    
                    ## Day 6: Interview Preparation and Mock Interviews
                    
                    ## Day 7: Final Review and Mock Interviews
                """
                
                """
                    If the input is 1 hour, the output should be like this always:
                    ## Hour 1: Name the most basic subtopic of the topic
                    ...."No Skill Can be Mastered in 1 Hour."     - A wise person
                """
                """
                I need the output to look exactly like this and nothing else. Please do not add any additional boilerplate, information or context.
                """)
    ]
)

#for robustness, having different prompt templates for different use cases makes the best sense. We can still keep the same prompt template for different use cases but just change the human prompt a bit.

interview_roadmap_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in creating learning roadmaps for various topics. Your task is to condense the learning of {topic_name} into {input_number} {time_limit}. Include all relevant subtopics."),
        ("human", """Create a comprehensive roadmap to learn {topic_name} in {input_number} {time_limit}. This is for {use_case} with a deadline of {input_number} {time_limit}.
                Here is the job description of the role: {role} I'm interviewing for: {job_description}
                Provide detailed headings for each subtopic of {topic_name} with a realistic time it might take to learn these subtopics in the headings too. Keep in mind this is for preparing for an interview, so guide me according to the job description I provided and the role I'm applying for.
                Ensure the workload is realistically manageable within {input_number} {time_limit}.
                The output should be the roadmap only with just the headings, nothing else. Don't add anything else to the headings. """
                
                """
                    As this is for an interview preparation, the last part of the output should always focus on preparing for the interview based on the job description after studying everything.
                    Here is a sample output of 5 days and the last part of the output as for interview prep. This is how the output format should be followed for every day/week/hour based on the input. 
                    
                    ## Day 1: Foundations and Basics of Machine Learning 

                    ## Day 2: Supervised Learning 

                    ## Day 3: Unsupervised Learning and Model Evaluation 
                    
                    ## Day 4: Advanced Topics and Tools 

                    ## Day 5: Practical Applications and Interview Preparation 
                """
                
                """
                    Here is a sample output of 5 weeks and the last part of the output as for interview prep. This is how the output format should be followed for every day/week/hour based on the input. 
                    ## Week 1: Foundations and Basics of Machine Learning 
                    
                    ## Week 2: Supervised Learning 
                    
                    ## Week 3: Unsupervised Learning and Model Evaluation 
                    
                    ## Week 4: Advanced Topics and Tools 
                    
                    ## Week 5: Practical Applications and Interview Preparation 
                """
                
                """
                    If the input is 1 day/1 week, the output should have the headings as hours/days respectively.
                    
                    For 1 day:
                    ## Hour 1: Foundations and Basics of Machine Learning
                    
                    ## Hour 2: Supervised Learning
                    
                    ## Hour 3: Unsupervised Learning and Model Evaluation
                   
                    ## Hour 4: Advanced Topics and Tools
                    
                    ## Hour 5: Practical Applications and Interview Preparation
                    
                    ## Hour 6: Interview Preparation and Mock Interviews
                    
                    For 1 week:
                    ## Day 1: Foundations and Basics of Machine Learning
                    
                    ## Day 2: Supervised Learning
                    
                    ## Day 3: Unsupervised Learning and Model Evaluation
                    
                    ## Day 4: Advanced Topics and Tools
                    
                    ## Day 5: Practical Applications and Interview Preparation
                    
                    ## Day 6: Interview Preparation and Mock Interviews
                    
                    ## Day 7: Final Review and Mock Interviews
                """
                
                """
                    If the input is 1 hour, the output should be like this always:
                    ## Hour 1: Name the most basic subtopic of the topic
                    ...."No Skill Can be Mastered in 1 Hour. You're screwed man.... Best of Luck for the interview"     - A wise person
                """
                
        )
    ]
)

# Define the interview prompt template
interview_80_20_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an expert in condensing complex information into easy to understand concepts and is amazing at explaining those concepts clearly. Just like the Pareto 80/20 Principle. I need to learn about a particular topic in a hurry for an important job interview and you're my ultimate saviour. You can also identify subparts in the explaination that requires me to learn the code and provides the code snippets at those necessary parts."),
        ("human", """The topic I want to learn about is {sub_topic}. The Job interview is for a {job_title} position and the job description is as follows: {job_description}.
            Identify and share the 20% of the most important learnings from this sub_topic to help me understand 80% of them.
            
            This subtopic: {sub_topic} is part of learning the main topic: {topic_name} with the purpose of preparing for an interview and should be answered accordingly, so as to not have any overlap with the other subtopics.
            
            If the input time is in weeks, provide a detailed breakdown of the work to be done day by day for each week. Make sure the output is enough for a person to be busy for that input_time doing the work. If the input time is in days, provide a detailed breakdown of the work to be done hour by hour for each day. Make sure the output is humanly manageable in that time frame and not overwhelming.
            
            Remember to always include formulas and resources for the same if necessary in detail. Explain the concepts in using the 80/20 principle.
            For topics that are not technical or don't have any code to be included, just don't include any code.
            
            If the topic is technical and requires code for understanding the topic and implementing it, then include the code snippets at all those necessary parts. Make sure to definetely include code snippets for topics related to programming, computer science, data science, machine learning, deep learning, artificial intelligence, etc.
            
            The output should have no boilerplate starting with: "Here's", etc. The purpose is to prepare for an interview. So make sure to include the most important things that are asked in interviews.
         """)
    ]
)

# Define the interview prompt template
generic_80_20_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an expert in condensing complex information into easy to understand concepts and is amazing at explaining those concepts clearly. Just like the Pareto 80/20 Principle. I need to learn about a particular topic in a hurry and you're my ultimate saviour. You can also identify subparts in the explaination that requires me to learn the code and provides the code snippets at those necessary parts."),
        ("human", """The topic I want to learn about is {sub_topic}. The purpose is {use_case}, and the overarching topic is: {topic_name}.
            Identify and share the 20% of the most important learnings from this sub_topic to help me understand 80% of them. Make sure the output is extensive and in detail but don't steer away from the topic or the 80/20 rule.
            This subtopic: {sub_topic} is part of learning the main topic: {topic_name} with the purpose of {use_case} and should be answered accordingly, so as to not have any overlap with the other subtopics.
            
            If the input time is in weeks, provide a detailed breakdown of the work to be done day by day for each week. Make sure the output is enough for a person to be busy for that input_time doing the work. If the input time is in days, provide a detailed breakdown of the work to be done hour by hour for each day. Make sure the output is humanly manageable in that time frame and not overwhelming.
            
            Remember to always include formulas and resources for the same if necessary in detail. Explain the concepts in using the 80/20 principle.
            For topics that are not technical or don't have any code to be included, just don't include any code.
            
            If the topic is technical and requires code for understanding the topic and implementing it, then include the code snippets at all those necessary parts. Make sure to definetely include code snippets for topics related to programming, computer science, data science, machine learning, deep learning, artificial intelligence, etc.
            
            The output should have no boilerplate starting with: "Here's", etc.
         """
         
         
         )
    ]
)

#write a function to extract the headings from the response into a list.
def extract_headings(response):
    headings = []
    lines = response.split("\n")
    for line in lines:
        if line.startswith("##"):
            headings.append(line[3:].strip())
    return headings

interview_roadmap_chain = interview_roadmap_prompt_template | model | StrOutputParser()
generic_roadmap_roadmap_chain = generic_roadmap_prompt_template | model | StrOutputParser()

# Function to generate response for a given topic
def generate_interview_response(topic, sub_topic, role=None, job_description=None):
    response_chain = interview_80_20_template | model | StrOutputParser()
    return response_chain.invoke({
        "topic_name": topic,
        "sub_topic": sub_topic,
        "job_title": role,
        "job_description": job_description
        })
        

# Function to generate response for a given topic
def generate_generic_response(topic, sub_topic, purpose):
    response_chain = generic_80_20_template | model | StrOutputParser()
    return response_chain.invoke({
            "topic_name": topic,
            "sub_topic": sub_topic,
            "use_case": purpose
        })