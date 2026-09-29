import os

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# VALIDATE ENVIRONMENT VARIABLES
# ============================================================

if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY is missing from your .env file.")
    st.stop()

if not PINECONE_INDEX_NAME:
    st.error("PINECONE_INDEX_NAME is missing from your .env file.")
    st.stop()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing from your .env file.")
    st.stop()


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")

st.write(
    "Ask questions about the documents stored in the "
    "Pinecone vector database."
)


# ============================================================
# INITIALIZE PINECONE
# ============================================================

try:

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

except Exception as e:

    st.error(
        f"Failed to connect to Pinecone: {e}"
    )

    st.stop()


# ============================================================
# INITIALIZE HUGGINGFACE EMBEDDINGS
# ============================================================

try:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

except Exception as e:

    st.error(
        f"Failed to load embedding model: {e}"
    )

    st.stop()


# ============================================================
# INITIALIZE VECTOR STORE
# ============================================================

try:

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

except Exception as e:

    st.error(
        f"Failed to initialize Pinecone vector store: {e}"
    )

    st.stop()


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if prompt:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        HumanMessage(
            content=prompt
        )
    )


    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    try:

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5
            }
        )

        docs = retriever.invoke(prompt)

    except Exception as e:

        st.error(
            f"Error while retrieving documents: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    if docs:

        docs_text = "\n\n---\n\n".join(
            doc.page_content
            for doc in docs
        )

    else:

        docs_text = (
            "No relevant documents were found."
        )


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = f"""
You are an assistant for question-answering tasks.

Use the retrieved context below to answer the user's question.

Rules:

1. Answer only using the retrieved context.
2. If the answer is not available in the context,
   say "I don't know based on the provided documents."
3. Do not invent information.
4. Keep the answer concise.
5. Use a maximum of three sentences.

Retrieved context:

{docs_text}
"""


    # --------------------------------------------------------
    # INITIALIZE GROQ LLM
    # --------------------------------------------------------

    try:

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="openai/gpt-oss-120b",
            temperature=0.2
        )

    except Exception as e:

        st.error(
            f"Failed to initialize Groq: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE MESSAGE HISTORY
    # --------------------------------------------------------

    messages = [
        (
            "system",
            system_prompt
        )
    ]


    # Add previous conversation
    for message in st.session_state.messages:

        if isinstance(message, HumanMessage):

            messages.append(
                (
                    "human",
                    message.content
                )
            )

        elif isinstance(message, AIMessage):

            messages.append(
                (
                    "assistant",
                    message.content
                )
            )


    # --------------------------------------------------------
    # INVOKE LLM
    # --------------------------------------------------------

    try:

        response = llm.invoke(
            messages
        )

        result = response.content

    except Exception as e:

        st.error(
            f"Error while generating response: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(result)


    # --------------------------------------------------------
    # SAVE RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        AIMessage(
            content=result
        )
    )