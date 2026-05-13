
import pandas as pd
import chainlit as cl
from pypdf import PdfReader

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = 'qwen2.5-coder:7b'
)

response = llm.invoke("What is RAG?")

print(response.content)

pdf_text = ""

@cl.step(type="tool")
async def tool():
    # Fake tool
    await cl.sleep(2)
    return "Response from the tool!"

@cl.on_chat_start
async def start():

    global pdf_text

    files = await cl.AskFileMessage(
        content="Please upload a PDF file",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180
    ).send()

    file = files[0]

    reader = PdfReader(file.path)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    pdf_text = text

    print(text[:1000])

    await cl.Message(
        content=f"PDF uploaded successfully!\n\nLength: {len(pdf_text)} characters"
    ).send()

@cl.on_message
async def main(message: cl.Message):

    global pdf_text

    prompt = f"""
    Answer the question based on the PDF content below.

    PDF Content:
    {pdf_text[:4000]}

    Question:
    {message.content}
    """

    response = llm.invoke(prompt)

    await cl.Message(
        content=response.content
    ).send()

