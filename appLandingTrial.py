from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama

load_dotenv()

# LLama3_2 = ChatOllama(model= "llama3.2:3b-instruct-q2_K")
# LLama3_1 = ChatOllama(model= "llama3.1:8b-instruct-q3_K_L")
openAILLM = ChatOpenAI(model="gpt-4o")

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

#write a function to extract the headings from the response into a list.
# Define the function to extract headings using RunnableLambda
extract_headings_lambda = RunnableLambda(lambda response: [
    line[2:].strip() for line in response.split("\n") if line.startswith("##")
])

interview_roadmap_chain = interview_roadmap_prompt_template | openAILLM | StrOutputParser() | extract_headings_lambda
generic_roadmap_prompt_template = generic_roadmap_prompt_template | openAILLM | StrOutputParser() | extract_headings_lambda

interiew_response = interview_roadmap_chain.invoke({
    "topic_name": "3-D Printing",
    "input_number": 1,
    "time_limit": "week",
    "use_case": "Interview Preparation",
    "role": None,
    "job_description": None
})

# print(interiew_response)

generic_response = generic_roadmap_prompt_template.invoke({
    "topic_name": "Machine Learning",
    "input_number": 2,
    "time_limit": "Weeks",
    "use_case": "Making a project"
})


print(generic_response)
print("\n")
print(interiew_response)
print("\n")