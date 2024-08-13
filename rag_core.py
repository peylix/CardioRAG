# LangChain 接入Yuan模型
from modelscope import snapshot_download
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 调用模型
from LLM import Yuan2_LLM
from sentence_transformers import SentenceTransformer
# 向量化
from transformers.utils import is_torch_cuda_available, is_torch_mps_available
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser


def get_model():
    model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./model/autodl-fs')
    llm = Yuan2_LLM('model/autodl-fs/IEITYuan/Yuan2-2B-Mars-hf')
    return llm

def get_docs(directory):
    loader = TextLoader(directory)
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
    # 指定模型名称
    model_name = "moka-ai/m3e-base"

    # 创建模型实例并将模型保存到指定路径
    model = SentenceTransformer(model_name)
    model.save('model\\m3e-base')


    # 词嵌入模型
    EMBEDDING_DEVICE = "cuda" if is_torch_cuda_available() else "mps" if is_torch_mps_available() else "cpu"
    embeddings_model = HuggingFaceEmbeddings(
        model_name='model\m3e-base', 
        model_kwargs={'device': EMBEDDING_DEVICE}
    )

    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)

    return vectorstore


def get_metadata(vectorstore):
    return vectorstore.docstore._dict.values()

def retrieve_docs(query, vectorstore):

    # 向量查询
    query = '东方健康膳食模式'
    docs = vectorstore.similarity_search(query)
    print(docs[0].page_content)

    #向量检索
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents("东方健康膳食模式")
    
    return retriever, docs


def get_llm_response(question):
    unprocessed_docs = get_docs("demo_docs/demo.pdf") # TODO: Change to a more general form
    vectorstore = preprocess(unprocessed_docs)
    llm = get_model()
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

# system_template = "你是一名资深的医生，对于高血压有很深的经验"
# human_template = "请根据以下文档回答我的问题：\n\n{docs}\n\n问题：{question}"

# prompt = ChatPromptTemplate.from_messages([
#         ("system", system_template),
#         ("human" , human_template)
#     ])

# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser
# question = "高血压患者应该怎么饮食？"
# docs = retriever.get_relevant_documents(question)

# chain.invoke({"question": question, "docs": docs})