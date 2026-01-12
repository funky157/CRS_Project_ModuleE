# =====================================================
# CRS PROJECT
# File: chunker_v1_4_incremental.py
# Version: v1.4
# Purpose:
#   Incremental chunker – processes ONLY new topics
#
# Key Upgrade:
#   ✔ Prints newly added topic names clearly
#   ✔ Confirms chunk count per topic
#
# Author: Harish
# =====================================================

import os
import glob
import re
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -----------------------------------
# Load environment
# -----------------------------------
load_dotenv("../.env")

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("Missing GROQ_API_KEY in .env")

groq_client = Groq(api_key=GROQ_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

RAW_DIR = "db/raw/"
DB_DIR = "db/chroma/"
COLLECTION_NAME = "crs"

# -----------------------------------
# TEXT CLEANING
# -----------------------------------
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------------
# STAGE DETECTION
# -----------------------------------
def detect_stage(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["introduction", "overview", "basic idea"]):
        return "introduction"
    if any(k in t for k in ["is defined as", "definition", "can be defined"]):
        return "definition"
    if any(k in t for k in ["construction", "structure", "diagram"]):
        return "construction"
    if any(k in t for k in ["types of", "classification", "nmos", "pmos"]):
        return "types"
    if any(k in t for k in ["working", "operation", "principle"]):
        return "working"
    if any(k in t for k in ["formula", "equation"]):
        return "formulas"
    if any(k in t for k in ["characteristics", "curves"]):
        return "characteristics"
    if any(k in t for k in ["application", "used in"]):
        return "applications"
    if any(k in t for k in ["advantages", "benefits"]):
        return "advantages"
    if any(k in t for k in ["disadvantages", "limitations"]):
        return "disadvantages"

    return "general"

# -----------------------------------
# STAGE PROMPTS
# -----------------------------------
STAGE_PROMPTS = {
    "introduction": "Briefly introduce the topic and explain why it is important.",
    "definition": "Give a clear, exam-oriented definition.",
    "construction": "Explain construction or internal structure.",
    "types": "Explain different types or classifications.",
    "working": "Explain the working principle step by step.",
    "formulas": "Explain important formulas and terms.",
    "characteristics": "Explain important characteristics or curves.",
    "applications": "List practical applications.",
    "advantages": "Explain advantages.",
    "disadvantages": "Explain limitations.",
    "general": "Summarize the content concisely."
}

# -----------------------------------
# JUNK FILTER
# -----------------------------------
def is_junk_chunk(text: str) -> bool:
    if len(text) < 300:
        return True

    junk = [
        "references", "bibliography", "doi:",
        "isbn", "et al.", "journal"
    ]
    return any(j in text.lower() for j in junk)

# -----------------------------------
# LOAD RAW FILES
# -----------------------------------
def load_raw_files():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    topics = {}

    for f in files:
        topic = os.path.basename(f).replace("raw_", "").replace(".txt", "")
        with open(f, "r", encoding="utf-8") as file:
            text = file.read().strip()
            if text:
                topics[topic] = text

    return topics

# -----------------------------------
# CHUNK TEXT
# -----------------------------------
def chunk_text(text, size=1200):
    text = clean(text)
    return [text[i:i+size] for i in range(0, len(text), size)]

# -----------------------------------
# SUMMARIZE
# -----------------------------------
def summarize_chunk(chunk, stage):
    try:
        prompt = STAGE_PROMPTS.get(stage, "Summarize clearly.")
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception as e:
        print("[WARN] Summary skipped:", e)
        return "Summary skipped."

# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":
    print("\n===== CHUNKER v1.4 (INCREMENTAL + CONFIRMATION) =====\n")

    db = PersistentClient(path=DB_DIR)

    try:
        collection = db.get_collection(COLLECTION_NAME)
        print("[DB] Existing collection loaded.")
    except:
        collection = db.create_collection(COLLECTION_NAME)
        print("[DB] New collection created.")

    # Existing topics
    existing_topics = set()
    existing = collection.get(include=["metadatas"])
    for m in existing.get("metadatas", []):
        existing_topics.add(m["topic"])

    raw_topics = load_raw_files()
    newly_added = []

    for topic, text in raw_topics.items():
        if topic in existing_topics:
            print(f"[SKIP] {topic} already exists")
            continue

        print(f"\n[PROCESSING NEW TOPIC]")
        print(f"   Topic name : {topic}")
        print(f"   Source file: raw_{topic}.txt")

        chunks = chunk_text(text)
        added_chunks = 0

        for idx, chunk in enumerate(chunks):
            stage = detect_stage(chunk)

            if is_junk_chunk(chunk):
                summary = "Content skipped."
            else:
                summary = summarize_chunk(chunk, stage)

            emb = embedder.encode(chunk).tolist()

            collection.add(
                ids=[f"{topic}_{idx}"],
                embeddings=[emb],
                metadatas=[{
                    "topic": topic,
                    "stage": stage,
                    "summary": summary
                }]
            )

            added_chunks += 1

        print(f"   ✅ COMPLETED: {topic}")
        print(f"   ➕ Chunks added: {added_chunks}")
        newly_added.append(topic)

    # -----------------------------------
    # FINAL SUMMARY
    # -----------------------------------
    print("\n===== CHUNKER SUMMARY =====")

    if newly_added:
        print("New topics successfully added:")
        for t in newly_added:
            print(f"  • {t}")
    else:
        print("No new topics were added.")

    print("\n===== CHUNKER FINISHED =====\n")
