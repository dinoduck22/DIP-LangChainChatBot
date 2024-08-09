import streamlit as st
import os

st.title("API 설정")

st.markdown("""
OpenAI API 키 발급방법은 아래 링크를 참고 해주세요
* [발급방법](https://openai.com/index/openai-api/) 
""")

api_key = st.text_input("API 키 입력", type="password")
confirm_button = st.button("설정하기", key="api_key")

if confirm_button:
    # 환경변수 세팅
    os.environ["OPENAI_API_KEY"] = api_key
    # key =
    st.write(f"aPI 키가설정되었습니:다 '{api_key[:15]}*********'")

