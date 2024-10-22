import os
import jieba
import json

# LangChain 接入Yuan模型
from modelscope import snapshot_download
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 调用模型
from LLM import Yuan2_LLM
from sentence_transformers import SentenceTransformer
# 向量化
from transformers.utils import is_torch_cuda_available, is_torch_mps_available
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

def get_model():
    model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./model/autodl-fs')
    llm = Yuan2_LLM('model/autodl-fs/IEITYuan/Yuan2-2B-Mars-hf')
    return llm


def get_docs(directory):
    loader = PyPDFLoader(directory)
    documents = loader.load()

    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    docs = text_splitter.split_documents(documents)

    return docs


def preprocess(docs):
    # 加载模型
    model_name = "moka-ai/m3e-base"
    local_model_dir = "/mnt/workspace/m3e-base"

    # # 首先尝试从 Hugging Face 加载模型，如果失败则从本地加载
    # try:
    #     print("********** Trying to load the mode from Hugging Face **********")
    #     model = SentenceTransformer(model_name)
    #     model.save('model\\m3e-base')
    # except Exception as e:
    #     if os.path.exists(local_model_dir):
    #         print("********** Alas, failed to load the model from Hugging Face **********")
    #         print("********** Loading the model from local directory **********")
    #         model = SentenceTransformer(local_model_dir)
    #         model.save('model\\m3e-base')
    #     else:
    #         raise e

    # 创建模型实例并将模型保存到指定路径
    # model = SentenceTransformer(local_model_dir)
    # model.save('model\\m3e-base')

    #词嵌入模型
    EMBEDDING_DEVICE = "cuda" if is_torch_cuda_available() else "mps" if is_torch_mps_available() else "cpu"
    embeddings_model = HuggingFaceEmbeddings(
         model_name='model\m3e-base', 
         model_kwargs={'device': EMBEDDING_DEVICE}
    )

    # vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)

    """
    直接从文件内存中加载向量数据库
    """
    # 指定保存的路径
    save_path = "./faiss_index"

    # 加载向量数据库
    vectorstore = FAISS.load_local(save_path, embeddings_model, allow_dangerous_deserialization=True)
    return vectorstore


def get_metadata(vectorstore):
    return vectorstore.docstore._dict.values()


def expand_query(query, synonym_dict):
    words = jieba.lcut(query)
    expanded_query = []
    for word in words:
        if word in synonym_dict:
            expanded_query.extend(synonym_dict[word])
        else:
            expanded_query.append(word)
    return " ".join(expanded_query)


def reverse_query_expansion(query, synonym_dict):
    words = jieba.lcut(query)
    result = []
    for word in words:
        normalized_word = synonym_dict.get(word, word)
        result.append(normalized_word)
    return "".join(result)



def retrieve_docs(query, vectorstore):
    #向量检索
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    
    return retriever, docs


def get_llm_response(question):
    unprocessed_docs = get_docs("demo_docs/demo.pdf") # TODO: Change to a more general form
    vectorstore = preprocess(unprocessed_docs) # TODO: 预加载向量数据库
    llm = get_model()

    
    # TODO 在完善同义词词典后取消注释
    # with open("synonym_dict.json", "r", encoding="utf-8") as f:
    #     synonym_dict = json.load(f)
    # processed_query = reverse_query_expansion(question, synonym_dict)

    retriever, docs = retrieve_docs(question, vectorstore)

    system_template = "你是一名资深的医生，对于高血压有极为丰富的经验。"
    human_template = "请根据以下文档回答我的问题：\n\n{docs}\n\n问题：{question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human" , human_template)
    ])

    output_parser = StrOutputParser()

    docs = retriever.get_relevant_documents(question)
    formatted_prompt = prompt.format(docs=docs, question=question)

    llm_response = llm(formatted_prompt)
    parsed_response = output_parser.parse(llm_response)

    return parsed_response, docs


if __name__ == "__main__":
    question = "东方健康膳食模式"
    response, docs = get_llm_response(question)
    print(response)
    print(docs)