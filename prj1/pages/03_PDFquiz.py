from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_teddynote import logging
import streamlit as st
import os
from langchain_core.messages import ChatMessage

logging.langsmith("CH12-RAG")


def rag_setup(file_path, chunk_size=1000, chunk_overlap=50):
    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def create_quiz_chain(model_name="gpt-4o-mini", retriever=rag_setup("/home/hello/PycharmProjects/langChain/prj1/data/업종별 맞춤 AI 프롬프트 200선_이종범.pdf",300, 50)):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """당신은 초중고 학생을 가르치는 30년차 베테랑 교사입니다.
        당신의 임무는 주어진 문맥(context)을 활용하여 학생들을 위한 퀴즈(quiz)를 만드는 것입니다. 
        퀴즈는 4지선다형 객관식 문제로 만들어 주세요. 문항은 3문항을 만들어 주세요. 
        문항의 각 난이도는 쉬움, 보통, 어려움으로 나누어 주세요. 
        
        #문제 예시:
        
        질문(난이도)
        - 가) 보기1
        - 나) 보기2
        - 다) 보기3
        - 라) 보기4
        
        - 정답 :
        - 해설 : 정답인 이유에 대해서 자세히 설명해 주세요. 

    #Context: 
    {context} 

    #Answer:"""
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


# 대화 기록이 없다면, chat_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "pdf_quiz" not in st.session_state:
    st.session_state["pdf_quiz"] = []


# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))


# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)


# 캐시 디렉토리 구조
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 페이지 구성 #
st.title("PDF Quiz")

with st.sidebar:
    uploaded_file = st.file_uploader("PDF", type=["pdf"])


@st.cache_resource(show_spinner="trying...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    retriever = rag_setup(file_path, 2000, 50)
    return retriever


print_history()

if uploaded_file:
    retriever = embed_file(uploaded_file)
    st.session_state["pdf_quiz"] = create_quiz_chain("gpt-4o-mini", retriever)
else:
    st.warning("PDF upload needed")

question = st.chat_input("type in your question")
if question:
    st.chat_message("user").write(question)
    rag_chain = st.session_state["pdf_quiz"]
    ai_answer = rag_chain.invoke(question)
    st.chat_message("ai").write(ai_answer)
    add_history("user", question)
    add_history("ai", ai_answer)
