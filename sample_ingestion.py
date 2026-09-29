import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


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
# INITIALIZE PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# ============================================================
# INDEX NAME
# ============================================================

index_name = "langchain-sample-index-384"


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CHECK INDEX
# ============================================================

existing_indexes = [
    index_info["name"]
    for index_info in pc.list_indexes()
]


# ============================================================
# CREATE INDEX
# ============================================================

if index_name not in existing_indexes:

    print(
        f"Creating index: {index_name}"
    )

    pc.create_index(
        name=index_name,

        # MiniLM embedding dimension
        dimension=384,

        metric="cosine",

        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


    while True:

        status = pc.describe_index(
            index_name
        ).status

        if status["ready"]:
            break

        print(
            "Waiting for index..."
        )

        time.sleep(1)


# ============================================================
# CONNECT TO INDEX
# ============================================================

index = pc.Index(
    index_name
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# ============================================================
# SAMPLE DOCUMENTS
# ============================================================

document_1 = Document(
    page_content=(
        "I had chocolate chip pancakes "
        "and scrambled eggs for breakfast "
        "this morning."
    ),
    metadata={
        "source": "tweet"
    }
)


document_2 = Document(
    page_content=(
        "The weather forecast for tomorrow "
        "is cloudy and overcast, with a "
        "high of 62 degrees."
    ),
    metadata={
        "source": "news"
    }
)


document_3 = Document(
    page_content=(
        "Building an exciting new project "
        "with LangChain - come check it out!"
    ),
    metadata={
        "source": "tweet"
    }
)


document_4 = Document(
    page_content=(
        "Robbers broke into the city bank "
        "and stole $1 million in cash."
    ),
    metadata={
        "source": "news"
    }
)


document_5 = Document(
    page_content=(
        "Wow! That was an amazing movie. "
        "I can't wait to see it again."
    ),
    metadata={
        "source": "tweet"
    }
)


document_6 = Document(
    page_content=(
        "Is the new iPhone worth the price? "
        "Read this review to find out."
    ),
    metadata={
        "source": "website"
    }
)


document_7 = Document(
    page_content=(
        "The top 10 soccer players "
        "in the world right now."
    ),
    metadata={
        "source": "website"
    }
)


document_8 = Document(
    page_content=(
        "LangGraph is a framework for "
        "building stateful, agentic applications."
    ),
    metadata={
        "source": "tweet"
    }
)


document_9 = Document(
    page_content=(
        "The stock market is down 500 points "
        "today due to fears of a recession."
    ),
    metadata={
        "source": "news"
    }
)


document_10 = Document(
    page_content=(
        "I have a bad feeling I am going "
        "to get deleted."
    ),
    metadata={
        "source": "tweet"
    }
)


# ============================================================
# DOCUMENT LIST
# ============================================================

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10
]


# ============================================================
# GENERATE IDS
# ============================================================

uuids = [
    f"id{i}"
    for i in range(
        len(documents)
    )
]


# ============================================================
# INSERT DOCUMENTS
# ============================================================

print(
    "Uploading sample documents..."
)

vector_store.add_documents(
    documents=documents,
    ids=uuids
)


print(
    "Sample documents uploaded successfully."
)