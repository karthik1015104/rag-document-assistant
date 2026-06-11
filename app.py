
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import os

# Page config
st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Ask questions about your PDF — answers grounded in the document with page citations.")

# Load API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=GROQ_API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based strictly on the provided context.
Always cite the source by mentioning page numbers when available.
If the answer is not in the context, say: I could not find this in the document.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    return retriever, llm, prompt

def format_docs(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in docs
    )

def ask(question, retriever, llm, prompt):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": st.session_state.chat_history,
        "question": question
    })
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    return answer, docs

# Load chain
with st.spinner("Loading model and index..."):
    retriever, llm, prompt = load_chain()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your document..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(question, retriever, llm, prompt)
        st.markdown(answer)

        # Show sources
        with st.expander("📚 Sources"):
            for doc in sources:
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Page {page}:**")
                st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
