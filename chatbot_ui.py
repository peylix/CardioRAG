import streamlit as st
# from rag_core import get_llm_response
from logging_save import LoggingSave

with st.sidebar:
    st.image("cardiorag-logo-without-background.png", use_column_width=True)
    "🤖 该项目由 CardioRAG 设计并开发"
    "❤️ 我们衷心希望您身心健康"
    "🧑‍💻 [查看项目源代码](https://github.com/peylix/CardioRAG)"
    ""
    "*请注意，本应用不能完全代替专业医师*"

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

    st.chat_message("user").write(prompt)
    # response, ret_docs = get_llm_response(prompt)
    response, ret_docs = "Fake Answer", ["Fake Document 1", "Fake Document 2"]

    msg = response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": msg
        }
    )

    st.chat_message("assistant").write(msg)

    st.markdown("##### 🔍 相关文档")
    for doc in ret_docs:
        with st.expander(doc.metadata.get("source", "文档")):
            st.markdown(f"**页面编号:** {doc.metadata.get('page', '未知')}")
            st.markdown(f"**内容预览:**\n\n{doc.page_content[:500]}...")


# 显示表单
if "show_form" not in st.session_state:
    st.session_state.show_form = False

if st.session_state.show_form:
    with st.form(key="end_conversation_form"):
        # 大模型学习助手用户反馈调查
        st.subheader("大模型学习助手用户反馈调查")
        st.write("感谢您使用我们的“大模型学习助手”。为了帮助我们改进产品并更好地满足您的需求，请您花几分钟时间填写此简短的问卷。您的意见对我们非常重要！")

        # 功能满意度
        satisfaction_options = ["非常满意", "满意", "中等", "不满意", "非常不满意"]

        overall_satisfaction = st.radio("1. 您对“大模型学习助手”的整体满意度如何？", satisfaction_options, horizontal=True)
        response_speed_satisfaction = st.radio("2. 您对“大模型学习助手”的响应速度满意吗？", satisfaction_options, horizontal=True)
        answer_accuracy_satisfaction = st.radio("3. 您对“大模型学习助手”的答案准确性满意吗？", satisfaction_options, horizontal=True)
        interface_friendlyness_satisfaction = st.radio("4. 您对“大模型学习助手”的界面友好性满意吗？", satisfaction_options, horizontal=True)

        # 改进建议
        improvement_options = ["更快的回答速度", "更准确的答案", "更友好的用户界面", "更多的功能", "更个性化的服务", "其他（请说明）"]
        other_improvement = st.multiselect("1. 您认为“大模型学习助手”有哪些需要改进的地方？（可多选）", improvement_options[:-1])
        other_improvement_text = st.text_input("如果选择了“其他”，请在这里说明：")

        # 其他建议或反馈
        other_feedback = st.text_area("2. 您是否有其他建议或反馈？")

        # 提交按钮
        feedback_submit_button = st.form_submit_button("提交反馈")
        
        if feedback_submit_button:

            st.success("感谢您的反馈！")

            feedback_data = {
                'Overall Satisfaction': overall_satisfaction,
                'Response Speed Satisfaction': response_speed_satisfaction,
                'Answer Accuracy Satisfaction': answer_accuracy_satisfaction,
                'Interface Friendlyness Satisfaction': interface_friendlyness_satisfaction,
                'Improvement Suggestions': ', '.join(other_improvement),
                'Other Improvement Text': other_improvement_text,
                'Other Feedback': other_feedback
            }
            LoggingSave(feedback_data)
else:
    if st.button("结束对话",key = 'end_dialog_button'):
        st.session_state.messages.clear()  
        st.session_state.show_form = True  
        st.experimental_rerun() 
