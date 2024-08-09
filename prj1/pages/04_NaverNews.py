from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_teddynote import logging
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_community.document_loaders import WebBaseLoader
import bs4


logging.langsmith("CH12-RAG")


def news_rag_setup(url, chunk_size=1000, chunk_overlap=50):
    # 단계 1: 문서 로드(Load Documents)
    loader = WebBaseLoader(
        web_paths=(url, ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
            )
        ),
    )
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


def create_news_chain(retriever, model_name="gpt-4o-mini"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """너는 뉴스 기자야.
        사용자가 묻는 질문에 대답해줘. 

    #Question:
    {question}
    
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

if "news_chain" not in st.session_state:
    st.session_state["news_chain"] = []


# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))


# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)


# 페이지 구성 #
st.title("News")

with st.sidebar:
    news_url = st.text_input("url")
    confirm_btn = st.button("confirm")

if confirm_btn:
    retriever = news_rag_setup(news_url, chunk_size=2000, chunk_overlap=50)
    st.session_state["news_chain"] = create_news_chain(retriever)

print_history()

question = st.chat_input("type in your question")
if question:
    if st.session_state["news_chain"]:
        st.chat_message("user").write(question)
        rag_chain = st.session_state["news_chain"]
        ai_answer = rag_chain.invoke(question)
        st.chat_message("ai").write(ai_answer)
        add_history("user", question)
        add_history("ai", ai_answer)
    else:
        st.warning("put url")