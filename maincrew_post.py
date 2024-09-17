# Import Require Library
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from crewai import Agent
from crewai import Task
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Crew,Process

# LLM Monitering
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT']="Social Media Post AI Agent Moniter"

# Creating Web Page header
st.subheader("**Multi AI Agent Social Media Blog Post Generator...**")

# Getting Task From Web
with st.form(key='Query',clear_on_submit=True):
    post_content=st.text_input(label='**What Social Media Post Would you Like me to come up with Today?**')
    social_media=st.selectbox(label='**Select Social Media:**',
                              options=['Twitter','Facebook','Instagram','linkedin'],index=None)
    llm_model_name=st.selectbox(label='**Select LLM Model:**',
                              options=['Gemini Model','Lamma Model'],index=None)
    submit_button = st.form_submit_button('Submit.')
    if submit_button:
        st.info('Input Details...')
        st.markdown(f'Blog Post Name: {post_content} ...')
        st.markdown(f'Social Media Name: {social_media} ...')
        st.markdown(f'LLM Model Name: {llm_model_name} ...')


# Creating LLM Variable
def model_selection(value):
    if value == 'Gemini Model':
        os.environ['GOOGLE_API_KEY']=st.secrets['GOOGLE_API_KEY']
        llm_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',api_key=os.getenv('GOOGLE_API_KEY'))
        return llm_model
    else:
        os.environ['GROQ_API_KEY']=st.secrets['GROQ_API_KEY']
        llm_model = ChatGroq(model='llama3-8b-8192',api_key=os.getenv('GROQ_API_KEY'))
        return llm_model

LLM_Model=model_selection(llm_model_name)

# Creating Tools
os.environ['SERPER_API_KEY'] = st.secrets['SERPER_API_KEY']
search_tool = SerperDevTool()

# Creating a Multi AI Agents
web_reseacher = Agent(
    role='Senior Web Researcher',
    goal='Unccover ground breaking information of {input}',
    backstory='''you are Web Researcher with over a decade of experience, 
    adept at uncovering insights that drive strategic decisions. Renowned for 
    leading cross-functional teams and delivering actionable data, they have a proven 
    track record of navigating complex, identifying trends, and transforming raw 
    data into impactful business strategies.''',
    memory=True,
    tools=[search_tool],
    llm=LLM_Model
)

blog_writer = Agent(
    role='{social_media} Post Creator',
    goal='''You will create a {social_media} post about {input}. 
    observed by the Senior Web Researcher''',
    backstory=("""specializes in crafting {social_media} posts. 
               it leverages advanced industry insights into compelling 
               {social_media} posts for a professional audience."."""),
    memory=True,
    llm=LLM_Model,
    allow_delegation=False
) 

# Creating Task for Agents
reseacher_task = Task(
    description='''Find and summarize the latest and most relevant big trend in {input}, 
    focus on identifying pros and cons and the ovreall narrative. Gather insights from reputable sources, 
    including industry blogs, whitepapers, and expert opinions, to support the {social_media} blog post your final 
    report should clearly articulate the key points''',
    expected_output='come up with comprehensive short post on the latest trend {input}',
    agent=web_reseacher
)

writer_task = Task(
    description='''compose an insightful post on {input} on the basis of researcher task focus on latest trends and
    how it's impacting the industry. this content should be essy to understand, engaging and positive''',
    expected_output='write the intersting {social_media} post on {input}.',
    agent=blog_writer
)

# Creating Crew
crew = Crew(
    agents=[web_reseacher,blog_writer],
    tasks=[reseacher_task,writer_task],
    manager_llm=LLM_Model,
    process=Process.sequential,
    verbose=True
)

# Creating a Dict Input Variable 
inputs={
    'input':post_content,
    'social_media':social_media
}

# Query Answering
if st.button("Generate"):
    with st.spinner("Generate Response..."):
        res=crew.kickoff(inputs=inputs)
        result=str(res)
        st.info("Here is a Response..")
        st.markdown(res)
        st.download_button(label='Download Text File',
                           file_name=f'{post_content} post.txt',data=result)
