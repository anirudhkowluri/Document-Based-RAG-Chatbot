import os

from dotenv import load_dotenv
from pinecone import Pinecone

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)


# ============================================================
# VALIDATE ENVIRONMENT VARIABLES
# ============================================================

if not PINECONE_API_KEY:

    raise ValueError(
        "PINECONE_API_KEY is missing."
    )


if not PINECONE_INDEX_NAME:

    raise ValueError(
        "PINECONE_INDEX_NAME is missing."
    )


# ============================================================
# INITIALIZE PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


index = pc.Index(
    PINECONE_INDEX_NAME
)


# ============================================================
# INITIALIZE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# INITIALIZE VECTOR STORE
# ============================================================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",

    search_kwargs={
        "k": 5,
        "score_threshold": 0.6
    }
)


# ============================================================
# QUERY
# ============================================================

query = input(
    "Enter your question: "
)


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

results = retriever.invoke(
    query
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nRESULTS:\n")


if not results:

    print(
        "No relevant documents found."
    )

else:

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            result.page_content
        )

        print(
            "Metadata:",
            result.metadata
        )