import os
from pathlib import Path

# Language detection thresholds (share of Arabic letters in the text)
AR_RATIO_ENGLISH_MAX = 0.10  # below this -> English
AR_RATIO_ARABIC_MIN = 0.70  # above this -> Arabic, in between -> bilingual
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx"}

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "index"
OUTPUTS_DIR = ROOT / "outputs"
RUN_LOG_PATH = OUTPUTS_DIR / "run_logs.jsonl"
EVAL_CASES_PATH = ROOT / "tests" / "eval_cases.jsonl"

DOMAINS = ["hr", "it", "finance"]
LANGUAGES = ["en", "ar", "bilingual"]

# Models (team decision)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "atlas_docs"
EMBED_BATCH_SIZE = 16
# Retrieval
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
CANDIDATE_MULTIPLIER = 5  # how many extra candidates to fetch before diversifying
MAX_CHUNKS_PER_FILE = 2  # stops one document from filling every slot
ROUTER_CONFIDENCE_THRESHOLD = 0.6  # below this the router uses its fallback

# Graph limits
MAX_STEPS = 10
MAX_TOOL_CALLS = 3

NOT_FOUND_MESSAGE = "I couldn't find this in the company documents."
