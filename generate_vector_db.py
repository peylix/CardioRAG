import os
from dotenv import find_dotenv, load_dotenv
# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"]  # 填写控制台中获取的 APIKey 信息，也可以使用OPENAI_API
file_paths = []
folder_path = './cardioRAG' #我们放pdf文件的文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:10])

from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:

    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())
# print(texts)
for text in texts:
    # text = texts[1]
    # print(f"每一个元素的类型：{type(text)}.",
    #     f"该文档的描述性数据：{text.metadata}",
    #     f"查看该文档的内容:\n{text.page_content[0:]}",
    #     sep="\n------\n")
    import re

    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    text.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), text.page_content)
    # print(pdf_page.page_content)

    text.page_content = text.page_content.replace('•', '')
    text.page_content = text.page_content.replace(' ', '')
    # print(pdf_page.page_content)

    text.page_content = text.page_content.replace('\n\n', '\n')
    # print(text.page_content)

for i in file_paths[:10]:
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
    loader = PyMuPDFLoader(i)

    # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
    pdf_pages = loader.load()

    print(f"载入后的变量类型为：{type(pdf_pages)}，", f"该 PDF 一共包含 {len(pdf_pages)} 页")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)
# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用
from zhipuai_embedding import ZhipuAIEmbeddings

# 定义 Embeddings
# embedding = OpenAIEmbeddings()
embedding = ZhipuAIEmbeddings()
# embedding = QianfanEmbeddingsEndpoint()

# 定义持久化路径
persist_directory = './data_base/vector_db/chroma'

from langchain.vectorstores.chroma import Chroma

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)