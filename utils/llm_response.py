from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_ollama import ChatOllama

load_dotenv()

# LLama3_2 = ChatOllama(model= "llama3.2:3b-instruct-q2_K")
LLama3_1 = ChatOllama(model= "llama3.1:8b-instruct-q3_K_L")
# openAILLM = ChatOpenAI(model="gpt-4o")

roadmap_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an expert on creating roadmaps to learn complex or simple topics and condense the learning in {input_number} {time_limit}. You're the most knowledgable person for this job who covers every aspect of {topic_name} which I want to learn in the given timeframe. Also for that topic if there are any, include the main sub-sub-topics that are exteremely relevant to that subtopic. "),
        ("human", """Build me a roadmap to learn {topic_name}` in {input_number} days. I am doing this for {use_case} and have a deadline of {input_number} {time_limit}.
                Your job is to give me a comprehensive roadmap that covers every aspect important for this {topic_name}. Make sure to give me proper headings of the subtopics of {topic_name} I need to cover.
                Remember to only give me the amount of workload that I can do in {input_number} {time_limit} time. There's no need to give me stuff that will take more than {input_number} {time_limit} to learn.
                The output of this query should only be the roadmap and Nothing Else. Just the roadmap""")
    ]
)

roadmap_chain = roadmap_prompt_template | LLama3_1 | StrOutputParser()

response = roadmap_chain.invoke({
    "topic_name": "Machine Learning",
    "input_number": 5,
    "time_limit": "days",
    "use_case": "Interview Preparation"
})

# print(response)
