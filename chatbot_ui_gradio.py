import gradio as gr
from rag_core import get_llm_response

def rag_interface(question):
    llm_answer, documents = get_llm_response(question)
    # llm_answer = 'Fake Answer'
    # documents = ['Fake Document 1', 'Fake Document 2']
    
    # Format the documents for display
    document_text = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
    
    return llm_answer, document_text


interface = gr.Interface(
    fn=rag_interface,
    inputs="text",
    outputs=["text", "text"],
    title="知心智医🧑‍⚕️",
    description="你好！我是「知心智医」，你的私人医疗助手。我可以回答关于高血压及冠心病的问题。",
    examples=["动脉粥样硬化是如何形成的？", "冠心病的主要症状有哪些？"]
)


interface.launch()
