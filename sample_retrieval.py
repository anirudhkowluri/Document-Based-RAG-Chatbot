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


# ============================================================
# VALIDATE API KEY
# ============================================================

if not PINECONE_API_KEY:

    raise ValueError(
        "PINECONE_API_KEY is missing "
        "from the .env file."
    )


# ============================================================
# PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# ============================================================
# INDEX
# ============================================================

index_name = "langchain-sample-index-384"


index = pc.Index(
    index_name
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# ============================================================
# RETRIEVER
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
# RETRIEVE
# ============================================================

results = retriever.invoke(
    query
)


# ============================================================
# DISPLAY
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
            f"--- Result {i} ---"
        )

        print(
            result.page_content
        )

        print(
            "Metadata:",
            result.metadata
        )

        print()