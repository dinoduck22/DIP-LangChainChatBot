import os

import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from Cnvst_chain import EnglishConversationChain, SummaryChain


st.title("my GPT")

if ("OPENAI_API_KEY") in os.environ:
    st.write("API 키가 설정되었습니다.")

with st.sidebar:
    select_model = st.selectbox("Select model", ["gpt-4o-mini", "gpt-4o"], index=0)

# API KEY 정보 로드
load_dotenv()

# 프로젝트 이름
logging.langsmith("CH01-Basic")

# input area
situation = st.chat_input("type in the example of situation")


# 답변 생성
def generate_answer(chain_name, question):
    # 답변 생성 공간
    answer_container = st.empty()

    embed_chain = chain_name(model_name=select_model).create_chain()
    answer = embed_chain.stream({"question": question})

    final_answer = " "
    for token in answer:
        final_answer += token
        answer_container.markdown(final_answer)


# 답변 생성
if situation:
    # 사용자가 먼저 situation을 입력
    st.chat_message("user").write(situation)

    # AI 답변
    with st.chat_message("ai"):
        generate_answer(SummaryChain, situation)

