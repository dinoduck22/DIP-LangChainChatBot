import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_teddynote import logging
import os


def rag_retrieve_setup(file_path, chunk_size=1000, chunk_overlap=50, k=4, weight=0.5):
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
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(
        split_documents,
    )
    sparse_retriever.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[weight, (1-weight)],
        search_type="mmr",
    )

    return ensemble_retriever


# 캐시 디렉토리 구조
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if "my_retriever" not in st.session_state:
    st.session_state["my_retriever"] = None


# 페이지 구성 #
st.title("Search Analysis")

with st.sidebar:
    uploaded_file = st.file_uploader("PDF", type=["pdf"])

col1, col2 = st.columns(2)

search_k = col1.number_input("number of search document", min_value=1, max_value=10, value=4, step=1)

retriever_weight = col1.slider("Retriever 가중치 (높음:의미검색 낮음:키워드검색", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

chunk_size = col2.number_input("chunk_size", min_value=100, max_value=2000, value=500, step=100)

chunk_overlap = col2.number_input("chunk_overlap", min_value=0, max_value=200, value=50, step=10)

confirm_btn = st.button("confirm")


@st.cache_resource(show_spinner="trying...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


if confirm_btn:
    if uploaded_file:
        file_path = embed_file(uploaded_file)
        st.session_state["my_retriever"] = rag_retrieve_setup(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=search_k,
            weight=retriever_weight,
        )
        st.write("setting completed")

# chatting
user_input = st.chat_input("please input search content")

if user_input:
    retriever = st.session_state["my_retriever"]
    st.chat_message("user").write(user_input)
    if retriever:
        # result confirm
        searched_docs = retriever.invoke(user_input)
        # list[documents]
        # ai chatting container
        with st.chat_message("ai"):
            for i, doc in enumerate(searched_docs):
                st.markdown(f"[{i}] {doc.page_content}")

