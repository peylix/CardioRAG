from LLM import Yuan2_LLM

import os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def generate_response(input_text):
    # llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    llm = Yuan2_LLM('model/autodl-fs/IEITYuan/Yuan2-2B-Mars-hf')
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    # st.info(output)
    return output


def get_vectordb():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    #      # 加载模型
    #     local_model_dir = "model/m3e-base"

    #     # 词嵌入模型
    #     EMBEDDING_DEVICE = "cuda" if is_torch_cuda_available() else "mps" if is_torch_mps_available() else "cpu"
    #     embedding = HuggingFaceEmbeddings(
    #         model_name=local_model_dir,
    #         model_kwargs={'device': EMBEDDING_DEVICE}
    #     )
    # 向量数据库持久化路径
    persist_directory = './data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    return vectordb


# 带有历史记录的问答链
def get_chat_qa_chain(question: str):
    vectordb = get_vectordb()
    # llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key)
    llm = Yuan2_LLM('model/autodl-fs/IEITYuan/Yuan2-2B-Mars-hf')
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']


# 不带历史记录的问答链
def get_qa_chain(question: str):
    vectordb = get_vectordb()
    # llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0,openai_api_key = openai_api_key)
    llm = Yuan2_LLM('model/autodl-fs/IEITYuan/Yuan2-2B-Mars-hf')
    template = """使用以下上下文来回答最后的问题。如果是英文，就按单词分割使用用医学语言风格翻译后再用中文回答，如果你不知道答案，
    就说你不知道，不要试图编造答案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 知心知医AI助手')
    # openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    # selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=450)
    if prompt := st.chat_input("Say something"):
        if False:
            st.info("出现了一些错误，请稍后再试。")
            st.stop()
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])

        """想搞的，不好搞，ui大佬搞一下"""
        # st.markdown("##### 🔍 相关文档")
        # for doc in ret_docs:
        #     with st.expander(doc.metadata.get("source", "文档")):
        #         st.markdown(f"**页面编号:** {doc.metadata.get('page', '未知')}")
        #         st.markdown(f"**内容预览:**\n\n{doc.page_content[:500]}...")


if __name__ == "__main__":
    main()