import os
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_huggingface import HuggingFaceEndpoint
from retrieval.store import vector_store_manager

# Simple in-memory store; replace with RedisChatMessageHistory for production
_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}\nSOURCE: {d.metadata.get('source')}"
                       for i, d in enumerate(docs))

def build_chain():
    retriever = vector_store_manager.get_retriever({"k": 4})

    system = (
        "You are DocChat RAG, a helpful assistant. "
        "Answer strictly from the provided context. "
        "If the answer is not in context, say you don't know."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {input}\n\nContext:\n{context}\n\nAnswer concisely with sources when relevant.")
    ])

    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_CHAT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        max_new_tokens=512,
        temperature=0.2,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    base = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with_history = RunnableWithMessageHistory(
        base,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return with_history
