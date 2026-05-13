import chainlit as cl
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


#launching models
llm = ChatOllama(
    model = 'qwen2.5-coder:7b'
)

#global attribute pdf_text
pdf_text = ""
vectorstore = None
chunks = 0

embeddings = OllamaEmbeddings(
    model = 'nomic-embed-text'
)

def parse_pdf(filename):
    reader = PdfReader(filename)

    text = ""

    for i,page in enumerate(reader.pages):
        page_text = page.extract_text() or ""

        text += f"\n\n===== PAGE {i + 1} =====\n"
        text += page_text   

    return text

def chunk_text(pdf_text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 200
    )

    chunks = splitter.split_text(pdf_text)

    return chunks

def called_prompt(question, k):
    results = vectorstore.similarity_search(question, k = k)

    context = "\n\n".join([
        doc.page_content for doc in results
    ])

    prompt = f"""
        Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {question}

        If the answer is not in the context, say you don't know.
    """

    return prompt


@cl.on_message
async def main(message: cl.Message):

    # Case 1: User uploads PDF
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":

                pdf_text = parse_pdf(element.path)
                chunks = chunk_text(pdf_text)
                
                if(chunks):
                    vectorstore = FAISS.from_texts(
                        chunks,
                        embedding= embeddings
                    )

                    # set_chunks & vectorstore for reuse purposes
                    cl.user_session.set('pdf_text', pdf_text)
                    cl.user_session.set("chunks", chunks)
                    cl.user_session.set("vectorstore", vectorstore)

                    # in cases where users upload PDF and yield questions simultaneously
                    if message.content.strip():
                        question = message.content
                        prompt = called_prompt(question, len(chunks))

                        response = llm.invoke(prompt)

                        await cl.Message(
                            content = response.content
                        ).send()

                    else:
                        await cl.Message(
                            content = "Now ask me a question about the PDF."
                        ).send()

                return  

    # Case 2: User asks a question
    if vectorstore is None or chunks is None:
        await cl.Message(
            content = 'Please upload a PDF first'
        ).send()
        return
    
    prompt = called_prompt(message.content, len(chunks))

    response = llm.invoke(prompt)

    await cl.Message(
        content=response.content
    ).send()