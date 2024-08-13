import streamlit as st
from rag_core import get_llm_response


with st.sidebar:
    "🤖 该项目由 CardioRAG 设计并开发"
    "❤️ 我们衷心希望您身心健康"
    "🧑‍💻 [查看项目源代码](https://github.com/peylix/CardioRAG)"

st.title("知心智医 🧑‍⚕️")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "你好！我是「知心智医」，你的私人医疗助手。我可以回答关于高血压及冠心病的问题。"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if False:
        st.info("出现了一些错误，请稍后再试。")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    st.chat_message("您").write(prompt)
    response, ret_docs = get_llm_response(prompt)

    msg = response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": msg
        }
    )

    st.chat_message("智医").write(msg)