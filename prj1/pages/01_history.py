import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage

# title
st.title("memory-chatbot")

# 프로젝트 이름
logging.langsmith("CH05-Memory")

# 세션 기록을 저장할 딕셔너리
if "store" not in st.session_state:
    st.session_state["store"] = {}


# 새로 고침 해도 저장 가능 하게 변수(list)에 저장 함
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# 대화 내용 저장
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))


# 이전 대화 내용 출력
def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메세지 출력
        st.chat_message(chat_message.role).write(chat_message.content)


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 체인 생성
def create_chain(model_name="gpt-4o-mini"):
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요."),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name=model_name)

    # 일반 Chain 생성
    base_chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,  # 세션 기록을 가져 오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history


# 페이지 구성 #
with st.sidebar:
    select_model = st.selectbox("Select model", ["gpt-4o-mini", "gpt-4o"], index=0)
    session_id = st.text_input("put your session id")

# 이전 대화 내용 출력
print_history()

# 대화 입력
question = st.chat_input("type in question")
if question:
    st.chat_message("user").write(question)

    chain = create_chain()

    answer = chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

    with st.chat_message("ai"):
        chat_container = st.empty()
        ai_answer = " "
        for token in answer:
            ai_answer += token
            chat_container.markdown(ai_answer)

    add_history("user", question)
    add_history("ai", ai_answer)
