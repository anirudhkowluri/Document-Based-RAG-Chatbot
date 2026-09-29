import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        "PINECONE_API_KEY is missing from the .env file."
    )


if not PINECONE_INDEX_NAME:

    raise ValueError(
        "PINECONE_INDEX_NAME is missing from the .env file."
    )


# ============================================================
# INITIALIZE PINECONE
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# ============================================================
# INITIALIZE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CHECK EXISTING INDEXES
# ============================================================

existing_indexes = [
    index_info["name"]
    for index_info in pc.list_indexes()
]


# ============================================================
# CREATE INDEX IF REQUIRED
# ============================================================

if PINECONE_INDEX_NAME not in existing_indexes:

    print(
        f"Creating Pinecone index: "
        f"{PINECONE_INDEX_NAME}"
    )

    pc.create_index(
        name=PINECONE_INDEX_NAME,

        # all-MiniLM-L6-v2 = 384 dimensions
        dimension=384,

        metric="cosine",

        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


    # --------------------------------------------------------
    # Wait until index is ready
    # --------------------------------------------------------

    while True:

        status = pc.describe_index(
            PINECONE_INDEX_NAME
        ).status

        if status["ready"]:
            break

        print(
            "Waiting for Pinecone index..."
        )

        time.sleep(1)


else:

    print(
        f"Pinecone index "
        f"'{PINECONE_INDEX_NAME}' already exists."
    )


# ============================================================
# CONNECT TO INDEX
# ============================================================

index = pc.Index(
    PINECONE_INDEX_NAME
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

documents_path = "documents/"

if not os.path.exists(documents_path):

    raise FileNotFoundError(
        "The 'documents/' folder does not exist."
    )


print(
    "Loading PDF documents..."
)

loader = PyPDFDirectoryLoader(
    documents_path
)

raw_documents = loader.load()


if not raw_documents:

    raise ValueError(
        "No PDF documents were found "
        "inside the documents/ folder."
    )


print(
    f"Loaded {len(raw_documents)} PDF pages."
)


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

print(
    "Splitting documents..."
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False
)


documents = text_splitter.split_documents(
    raw_documents
)


print(
    f"Created {len(documents)} document chunks."
)


# ============================================================
# GENERATE UNIQUE IDS
# ============================================================

uuids = [
    f"id{i}"
    for i in range(
        len(documents)
    )
]


# ============================================================
# ADD DOCUMENTS TO PINECONE
# ============================================================

print(
    "Uploading documents to Pinecone..."
)

vector_store.add_documents(
    documents=documents,
    ids=uuids
)


print(
    "Documents successfully added to Pinecone."
)