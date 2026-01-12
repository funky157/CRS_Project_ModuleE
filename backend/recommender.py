# =====================================================
# CRS PROJECT
# File: recommender.py
# Version: v3.5
# Purpose:
#   Structured, time-aware concept explainer
#   with ranked related-topic discovery
#
# Author: Harish
# =====================================================




print(">>> recommender_v3.5 CONCEPT EXPLAINER LOADED <<<")

import re
import random
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db", "chroma")

COLLECTION_NAME = "crs"

WORDS_PER_MIN = 120
TOP_K_EXPLANATION = 30
TOP_K_RELATED_SEARCH = 40
MAX_RELATED_TILES = 12   # 🔥 exactly what you asked

STAGE_ORDER = [
    "definition", "types", "construction", "working",
    "formulas", "applications", "advantages",
    "disadvantages", "general"
]

STAGE_TITLES = {
    "definition": "Definition",
    "types": "Types",
    "construction": "Construction / Structure",
    "working": "Working Principle",
    "formulas": "Key Formulas",
    "applications": "Applications",
    "advantages": "Advantages",
    "disadvantages": "Limitations",
    "general": "Additional Notes"
}

STAGE_WEIGHT = {
    "definition": 0.12,
    "types": 0.12,
    "construction": 0.14,
    "working": 0.20,
    "formulas": 0.14,
    "applications": 0.14,
    "advantages": 0.07,
    "disadvantages": 0.07,
    "general": 0.10
}

# -----------------------------------------------------
# LOAD MODELS & DB
# -----------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = PersistentClient(path=DB_DIR)

try:
    collection = db.get_collection(COLLECTION_NAME)
    print("[INFO] Chroma collection 'crs' loaded.")
except Exception:
    collection = db.create_collection(COLLECTION_NAME)
    print("[WARN] Chroma collection 'crs' not found. Empty collection created.")


# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def embed_query(query: str):
    return embedder.encode(query).tolist()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    garbage = [
        "here's a", "here is a", "4-5 line",
        "for revision", "summary", "unfortunately",
        "the given text", "i can provide"
    ]
    for g in garbage:
        text = text.replace(g, "")

    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().capitalize()


def normalize(s: str):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

# -----------------------------------------------------
# EXPLANATION (UNCHANGED CORE)
# -----------------------------------------------------
def get_explanation(query: str, time_minutes: int):
    max_words = time_minutes * WORDS_PER_MIN
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_EXPLANATION,
        include=["metadatas"]
    )

    metadatas = results.get("metadatas", [[]])[0]

    sections = {k: [] for k in STAGE_ORDER}
    seen = set()
    used_words = 0

    for meta in metadatas:
        stage = meta.get("stage", "general")
        summary = clean_text(meta.get("summary", ""))

        if not summary:
            continue

        key = normalize(summary)
        if key in seen:
            continue
        seen.add(key)

        sections.setdefault(stage, []).append(summary)

    output = []

    for stage in STAGE_ORDER:
        lines = sections.get(stage, [])
        if not lines:
            continue

        stage_budget = int(max_words * STAGE_WEIGHT.get(stage, 0.1))
        stage_words = 0
        paragraph = []

        for line in lines:
            paragraph.append(line)
            w = len(line.split())
            stage_words += w
            used_words += w

            if stage_words >= stage_budget or used_words >= max_words:
                break

        if paragraph:
            output.append(f"\n{STAGE_TITLES[stage]}:\n")
            output.append(" ".join(paragraph) + "\n")

        if used_words >= max_words:
            break

    if not output:
        return f"{query} is an important electronics topic."

    return "\n".join(output).strip()

# -----------------------------------------------------
# RELATED TOPICS (RANKED → GENERIC → RANDOM)
# -----------------------------------------------------
def get_related_topics(query: str):
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K_RELATED_SEARCH,
        include=["metadatas", "distances"]
    )

    ranked = []
    seen = set()

    # 1️⃣ Most relevant (semantic)
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        topic = meta.get("topic", "")
        if not topic:
            continue

        topic_clean = topic.replace("_", " ").title()
        if topic_clean.lower() == query.lower():
            continue

        if topic_clean not in seen:
            ranked.append(topic_clean)
            seen.add(topic_clean)

        if len(ranked) >= MAX_RELATED_TILES // 2:
            break

    # 2️⃣ Random valid topics (exploration)
    all_topics = list(
        set(m["topic"] for m in results["metadatas"][0] if "topic" in m)
    )
    random.shuffle(all_topics)

    for t in all_topics:
        t_clean = t.replace("_", " ").title()
        if t_clean not in seen and t_clean.lower() != query.lower():
            ranked.append(t_clean)
            seen.add(t_clean)

        if len(ranked) >= MAX_RELATED_TILES:
            break

    return ranked

# -----------------------------------------------------
# PUBLIC API
# -----------------------------------------------------
def explain(query: str, time_minutes: int):
    return {
        "topic": query,
        "time_minutes": time_minutes,
        "explanation": get_explanation(query, time_minutes),
        "related_topics": get_related_topics(query)
    }
