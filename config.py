import os

# --- Server ---
HOST = "0.0.0.0"
PORT = 8000

# --- LLM ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_TEMPERATURE = 0.8
LLM_TIMEOUT = 120.0

# --- Admin ---
ADMIN_PIN = os.environ.get("ADMIN_PIN", "1234")

# --- Indexing ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "collections")

# --- Retrieval ---
BM25_TOP_K = 50
DENSE_TOP_K = 50
RRF_K = 60
MMR_TOP_K = 10
MMR_LAMBDA = 0.7
DEDUP_PENALTY = 0.4

# --- Sessions ---
HEARTBEAT_INTERVAL = 30
HEARTBEAT_TIMEOUT = 120

# --- Topics ---
TOPICS = [
    "Человечество — космическая цивилизация",
    "Неисчерпаемые блага космоса",
    "Тайны космоса в наших руках",
    "Земля — наш космический корабль",
]

TOPIC_PREFIXES = {
    "Человечество — космическая цивилизация": "ЛУНА",
    "Неисчерпаемые блага космоса": "ОРБИТА",
    "Тайны космоса в наших руках": "ЗВЕЗДА",
    "Земля — наш космический корабль": "ЗЕМЛЯ",
}

ROLE_SUFFIXES = {
    "optimist": "01",
    "pessimist": "02",
}

# --- Focus lenses for response diversity ---
FOCUS_LENSES = [
    "социальные последствия",
    "технологические прорывы",
    "этические дилеммы",
    "повседневная жизнь людей",
    "геополитические сдвиги",
    "неожиданные побочные эффекты",
    "культурные трансформации",
    "экологические аспекты",
]
