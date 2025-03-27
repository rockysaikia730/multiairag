import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

import re
import time

from typing import List,Annotated
from typing_extensions import TypedDict

import os
from dotenv import load_dotenv

import streamlit as st

import warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

#Load all API Keys
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("langsmith_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langgraph"
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

#Initialise connection with AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)

#Vectorise PDF and URLS
def process_pdfs(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pdf_loader = PyPDFLoader(file_path)
        docs.extend(pdf_loader.load())
        os.remove(file_path)
    return docs

def process_urls(url_list):
    docs = []
    for url in url_list:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
astra_vector_store = Cassandra(embedding=embeddings,
                               table_name="qa_mini_demo",
                               session=None,
                               keyspace=None)

# Section for PDF Upload
st.subheader("Upload PDFs")
uploaded_files = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type=["pdf"])

# Section for URL Input
st.subheader("Enter Website URLs")
url_list = st.text_area("Enter URLs (one per line)").split("\n")
url_list = [url.strip() for url in url_list if url.strip()]  # Clean up input

# Process files & URLs on button click
if st.button("Process Documents"):
    st.write("Processing...")

    all_docs = []
    
    if uploaded_files:
        all_docs.extend(process_pdfs(uploaded_files))
    
    if url_list:
        all_docs.extend(process_urls(url_list))
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    docs_split = text_splitter.split_documents(all_docs)

    astra_vector_store.add_documents(docs_split)

    st.success(f"✅ {len(docs_split)} document chunks added to the vector database!")

#retriever
retriever = astra_vector_store.as_retriever()

def retrieve(state):
  question = state["question"]
  documents = retriever.invoke(question)
  return {"documents":documents,"questions":question}

#Wiki_retriever
def wiki_search(state):
  wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
  wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
  question = state["question"]
  documents = wiki_tool.invoke(question)
  documents = Document(page_content=documents)
  return {"documents":[documents],"questions":question}

#Arxiv_retriever
def arxiv_search(state):
  arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
  arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
  question = state["question"]
  documents = arxiv_tool.invoke(question)
  documents = Document(page_content=documents)
  return {"documents":[documents],"questions":question}

#Routing
class RouteQuery(BaseModel):
  datasource:Literal["vectorstore","wiki_search","arxiv_search","deepseek"] = Field(...)

groq_api_key = os.getenv("groq_api_key")
llm = ChatGroq(groq_api_key=groq_api_key,model_name="Llama-3.3-70b-versatile")
llm1 = ChatGroq(groq_api_key=groq_api_key,model_name="deepseek-r1-distill-llama-70b")
structured_llm_router = llm1.with_structured_output(RouteQuery)

system="""You are an expert at routing a user question to a vectorstore, wikipedia, arxiv or llm.
Given the user's question, determine the most suitable information source:
Use Vectorstore If the question relates to well-indexed, structured knowledge available in the database.
Use Wikipedia If the question seeks general, well-established knowledge or historical context.
Use arXiv If the question concerns recent research trends, cutting-edge scientific developments, or novel theories.
Use LLM Response If the question requires reasoning, synthesis, or domain-specific insight that isn't readily available in the above sources.
"""
routeprompt = ChatPromptTemplate.from_messages([
    ("system",system),
    ("human","{question}")
])

question_router = routeprompt | structured_llm_router

def route_question(state):
  question = state["question"]
  source = question_router.invoke({"question":question})
  return source.datasource

class GraphState(TypedDict):
  question:str
  generation:str
  documents:List[str]

workflow = StateGraph(GraphState)
workflow.add_node("retrieve",retrieve)
workflow.add_node("wiki_search",wiki_search)
workflow.add_node("arxiv_search",arxiv_search)

workflow.add_conditional_edges(
  START,
  route_question,
  {
    "wiki_search":"wiki_search",
    "arxiv_search":"arxiv_search",
    "vectorstore":"retrieve",
    "deepseek":"deepseek",
  },
)

def deepseek(state):
   question = state["question"]
   prompt = ChatPromptTemplate.from_template(
        """Answer the following question. 
        Provide a detailed and accurate answer.
        Question: {question}
        Answer:"""
    )
   chain = prompt | llm1
   response = chain.invoke({"question": question})
   cleaned_text = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
   return {"generation": cleaned_text.strip()}

def chatbot(state):
  documents = state["documents"]
  question = state["question"]
  context = "\n\n".join([doc.page_content for doc in documents])

  prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context. 
        Think step by step and provide a detailed, accurate answer.
        Context:
        {context}
        Question: {question}
        Answer:"""
    )

  chain = prompt | llm
  response = chain.invoke({"context": context, "question": question})
  return {"generation": response.content}

workflow.add_node("chatbot",chatbot)
workflow.add_node("deepseek",deepseek)

workflow.add_edge("wiki_search", "chatbot")
workflow.add_edge("arxiv_search", "chatbot")
workflow.add_edge("retrieve", "chatbot")
workflow.add_edge("deepseek",END)
workflow.add_edge("chatbot",END)


app = workflow.compile()

def main():
    st.title("Your AI Tutor")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            spinner_placeholder = st.empty()
            
            start_time = time.time() 
            
            with st.spinner("Thinking..."):
                while True:
                    elapsed_time = int(time.time() - start_time)
                    spinner_placeholder.text(f"Thinking... ({elapsed_time}s elapsed)")
                    
                    inputs = {"question": prompt}
                    result = app.invoke(inputs)
                    
                    if result: 
                        break
                    time.sleep(0.5)  

                response = result["generation"]
                spinner_placeholder.empty()  
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
 

if __name__ == "__main__":
    main()

