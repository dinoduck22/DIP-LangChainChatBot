import streamlit as st
from langchain_core.messages import ChatMessage
from agent import create_agent

st.title("search chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "search_agent" not in st.session_state:
    st.session_state["search_agent"] = []


# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))


# 이전 까지의 대화를 출력
def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)


with st.sidebar:
    search_count = st.number_input("search result number", min_value=1, max_value=10, value=5, step=1)
    confirm_btn = st.button("confirm")

if confirm_btn:
    st.session_state["search_agent"] = create_agent(k=search_count)

print_history()

user_input = st.chat_input("type in your question")
if user_input:
    if st.session_state["search_agent"]:
        search_agent = st.session_state["search_agent"]
        st.chat_message("user").write(user_input)

        ai_answer = st.session_state["search_agent"].invoke({"input": user_input})
        st.chat_message("ai").write(ai_answer["output"])

        add_history("user", user_input)
        add_history("ai", ai_answer["output"])