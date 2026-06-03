import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Locale Configuration (Farsi) ---
LOCALE = "fa"
RESPONSE_LANGUAGE = "فارسی"
SOURCES_LABEL = "منابع"

# --- Model Configuration ---
# Persian dense embeddings; re-index all documents after changing this model.
DENSE_MODEL = "heydariAI/persian-embeddings"
SPARSE_MODEL = "Qdrant/bm25"
# FastEmbed BM25 has no Persian stemmer/stopwords — disable English stemming.
BM25_DISABLE_STEMMER = True
# Local Ollama model with strong multilingual + tool-calling support.
# Run: ollama pull qwen2.5:7b-instruct
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
LLM_TEMPERATURE = 0

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
# Slightly higher threshold — Persian text often tokenizes into more units.
BASE_TOKEN_THRESHOLD = 2500
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
# Character-based sizes tuned for Persian prose density.
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80
MIN_PARENT_SIZE = 1500
MAX_PARENT_SIZE = 3500
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]
TEXT_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", "؟ ", "! ", "؛ ", " ", ""]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
