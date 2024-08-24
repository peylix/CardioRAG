import streamlit as st
# from langchain_openai import ChatOpenAI
import os
import re
from LLM import Yuan2_LLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from transformers.utils import is_torch_cuda_available, is_torch_mps_available
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
from logging_save import LoggingSave
_ = load_dotenv(find_dotenv())    # read local .env file


#export OPENAI_API_KEY=
#os.environ["OPENAI_API_BASE"] = 'https://api.chatgptid.net/v1'
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
def has_more_than_one_chinese_question_mark(s):
    # 匹配中文问号
    pattern = r'？'
    # 查找所有匹配的中文问号
    matches = re.findall(pattern, s)
    # 判断匹配的数量是否大于1
    return len(matches) > 1

def clean_text(text):
    # 移除非文字内容
    cleaned_text = re.sub(r'\[图\d+\]', '', text)  # 移除形如 [图1] 的内容
    cleaned_text = re.sub(r'\[图\s*\d+\]', '', cleaned_text)  # 移除形如 [ 图 1 ] 的内容
    cleaned_text = re.sub(r'\[图\s*\d+\s*-\s*\d+\]', '', cleaned_text)  # 移除形如 [ 图 1 - 2 ] 的内容
    cleaned_text = re.sub(r'\[图\s*\d+\s*\.\s*\d+\]', '', cleaned_text)  # 移除形如 [ 图 1 . 2 ] 的内容
    cleaned_text = re.sub(r'\[参考图\s*\d+\]', '', cleaned_text)  # 移除形如 [题1] 的内容
    cleaned_text = re.sub(r'\[参考图\s*\d+\s*-\s*\d+\]', '', cleaned_text)  # 移除形如 [题1-2] 的内容
    cleaned_text = re.sub(r'\[参考图\s*\d+\s*\.\s*\d+\]', '', cleaned_text)  # 移除形如 [题1.2] 的内容
    cleaned_text = re.sub(r'\[表\s*\d+\]', '', cleaned_text)  # 移除形如 [题1] 的内容
    cleaned_text = re.sub(r'\[表\s*\d+\s*-\s*\d+\]', '', cleaned_text)  # 移除形如 [题1-2] 的内容
    cleaned_text = re.sub(r'\[表\s*\d+\s*\.\s*\d+\]', '', cleaned_text)  # 移除形如 [题1.2] 的内容
    cleaned_text = re.sub(r'\[见表\s*\d+\]', '', cleaned_text)  # 移除形如 [题1] 的内容
    cleaned_text = re.sub(r'\[见表\s*\d+\s*-\s*\d+\]', '', cleaned_text)  # 移除形如 [题1-2] 的内容
    cleaned_text = re.sub(r'\[见表\s*\d+\s*\.\s*\d+\]', '', cleaned_text)  # 移除形如 [题1.2] 的内容
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)  # 移除形如 [1] 的内容
    cleaned_text = re.sub(r'\[\d+\s*\-\s*\d+\]', '', cleaned_text)  # 移除形如 [1-2] 的内容
    cleaned_text = re.sub(r'\[\d+\s*\.\s*\d+\]', '', cleaned_text)  # 移除形如 [1.2] 的内容
    cleaned_text = re.sub(r'\[\d+\s*\.\s*\d+\.\d+\]', '', cleaned_text)  # 移除形如 [1.2] 的内容
    cleaned_text = re.sub(r'\[\·]', '', cleaned_text)  # 移除形如 [1.2] 的内容
    cleaned_text = re.sub(r'\[\·\s*\·]', '', cleaned_text)  # 移除形如 [1.2] 的内容
    
    # cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)  # 移除所有括号内的内容
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 移除多余的空格

    # 检查是否全为非中文字符
    if re.match(r'^[^\u4e00-\u9fff]*$', cleaned_text):
        cleaned_text = "请保重身体！"
    
    return cleaned_text

def get_first_part(text):
    truncated_text = text[:20]
    index_of_marker = truncated_text.find('、')
    if index_of_marker != -1:
        return truncated_text[:index_of_marker + 1]
    else:
        return truncated_text
    
def remove_trailing_character(s, char):
    return s.rstrip(char)

def generate_response(input_text):
    # llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    llm = Yuan2_LLM('IEITYuan/Yuan2-2B-Mars-hf')
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

def generate_rag(question:str):
    vectordb = get_vectordb()
    docs = vectordb.max_marginal_relevance_search(question,k=3)
    str="\n".join([clean_text(i.page_content) for i in docs])
    if str:
        return str
    return "抱歉，暂时未找到！"

def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    # 向量数据库持久化路径
    persist_directory = 'database/vectordb'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str):
    vectordb = get_vectordb()
    # llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key)
    llm = Yuan2_LLM('IEITYuan/Yuan2-2B-Mars-hf')
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str):
    vectordb = get_vectordb()
    # llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key)
    llm = Yuan2_LLM('IEITYuan/Yuan2-2B-Mars-hf')
    template = """你是一个高血压和冠心病专家，请使用以下上下文来回答最后的问题。如果是英文，就按单词分割使用用医学语言风格翻译后再用中文回答，如果你不知道答案，就说你不知道，不要试图编造答案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    # print('#'*60)
    # print(result["result"])
    # print(type(result["result"]))
    # print(len(result["result"]))
    # print('#'*60)
    flag=has_more_than_one_chinese_question_mark(result["result"])
    if len(result["result"]) < 500 and not flag:
        return result["result"]
    elif flag and len(result["result"]) < 500 :
        return "你是想问？\n\n"+result["result"]
    else:
        result = get_first_part(result["result"])
        return remove_trailing_character(result, '、')
    


# # Streamlit 应用程序界面
def main():
    import os

#     # 指定文件路径
#     file_path = 'model/IEITYuan/Yuan2-2B-Mars-hf'

#     # 检查文件是否存在
#     if not os.path.exists(file_path):
#         print("********** DownLoading the model to local directory **********")
#         from modelscope import snapshot_download
#         model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./model')
    with st.sidebar:
        st.image("cardiorag-logo-without-background.png", use_column_width=True)
        "🤖 该项目由 CardioRAG 设计并开发"
        "❤️ 我们衷心希望您身心健康"
        "🧑‍💻 [查看项目源代码](https://github.com/peylix/CardioRAG)"
        ""
        "*请注意，本应用不能完全代替专业医师*"
        ""
        "[复旦医院排行榜](https://fdygs.q-health.cn/news2022-1.aspx)"
    
    st.title("知心智医 🧑‍⚕️")
    # st.title('🦜🔗 知心知医AI助手')
    # openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    selected_method = st.sidebar.selectbox("选择模式", ["智能检索模式", "上下文检索问答模式", "普通模式", "检索模式"])
    # selected_method = st.radio(
    #     "你想选择哪种模式进行对话？",
    #     ["None", "qa_chain", "chat_qa_chain"],
    #     captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])
    

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "text": "你好！我是「知心智医」，你的私人医疗助手。我可以回答关于高血压及冠心病的问题。"
        }]

    messages = st.container(height=400)
    if prompt := st.chat_input("Say something"):
        if False:
            st.info("出现了一些错误，请稍后再试。")
            st.stop()
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})
        # answer = get_chat_qa_chain(prompt)
        if selected_method == "普通模式":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "智能检索模式":
            answer = get_qa_chain(prompt)
        elif selected_method == "检索模式":
            answer = generate_rag(prompt)
        elif selected_method == "上下文检索问答模式":
            answer = get_chat_qa_chain(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer.replace("<eod>", "")})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   

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
         
            # 使用st.cache_data来管理对话历史
            @st.cache_data(ttl=600)  # 缓存5分钟
            def cached_messages():
                return st.session_state.messages

            # 清除缓存
            cached_messages.clear() 


if __name__ == "__main__":
    main()