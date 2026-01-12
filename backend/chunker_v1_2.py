# =====================================================
# CRS PROJECT
# File: chunker_v1_2.py
# Version: v1.3 (SAFE IMPORT FIX)
# Purpose:
#   Chunk raw scraped content, generate stage-aware summaries,
#   detect learning stage, and store in vector DB
#
# Author: Harish
# =====================================================

import os
import glob
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -----------------------------------
# Optional LLM (Groq) import
# -----------------------------------
try:
    from groq import Groq
except ImportError:
    Groq = None

# -----------------------------------
# Load environment
# -----------------------------------
load_dotenv("../.env")

GROQ_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

if GROQ_KEY and Groq is not None:
    groq_client = Groq(api_key=GROQ_KEY)
else:
    print("[INFO] GROQ API not available. Summarization disabled.")

# -----------------------------------
# Models & paths
# -----------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

RAW_DIR = "db/raw/"
DB_DIR = "db/chroma/"

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
    if any(k in t for k in ["construction", "structure", "diagram", "cross section"]):
        return "construction"
    if any(k in t for k in ["types of", "classification", "nmos", "pmos"]):
        return "types"
    if any(k in t for k in ["working", "operation", "operates", "principle"]):
        return "working"
    if any(k in t for k in ["formula", "equation", "mathematical", "expression"]):
        return "formulas"
    if any(k in t for k in ["characteristics", "curves", "graph"]):
        return "characteristics"
    if any(k in t for k in ["application", "used in", "applications"]):
        return "applications"
    if any(k in t for k in ["advantages", "benefits"]):
        return "advantages"
    if any(k in t for k in ["disadvantages", "limitations"]):
        return "disadvantages"

    return "general"

# -----------------------------------
# STAGE-AWARE PROMPTS
# -----------------------------------
STAGE_PROMPTS = {
    "introduction": "Briefly introduce the topic and explain why it is important.",
    "definition": "Give a clear and concise definition in simple, exam-oriented language.",
    "construction": "Explain the physical construction or internal structure.",
    "types": "Explain different types or classifications.",
    "working": "Explain the working principle step-by-step.",
    "formulas": "Explain important formulas and what each term represents.",
    "characteristics": "Explain key characteristics or curves.",
    "applications": "Explain practical applications.",
    "advantages": "Explain advantages clearly.",
    "disadvantages": "Explain limitations clearly.",
    "general": "Summarize the content briefly."
}

# -----------------------------------
# JUNK CONTENT FILTER
# -----------------------------------
def is_junk_chunk(text: str) -> bool:
    t = text.lower()

    junk_keywords = [
        "references", "bibliography", "citation", "acknowledgements",
        "doi:", "isbn", "et al.", "journal", "vol.", "pp.", "copyright"
    ]

    if any(k in t for k in junk_keywords):
        return True

    if len(text) < 300:
        return True

    return False

# -----------------------------------
# LOAD RAW FILES
# -----------------------------------
def load_raw_text():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    data = {}

    for f in files:
        topic = os.path.basename(f).replace("raw_", "").replace(".txt", "")
        with open(f, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if content:
                data[topic] = content
            else:
                print(f"[WARN] Empty file skipped: {f}")

    print(f"[LOAD] Loaded {len(data)} topics")
    return data

# -----------------------------------
# CHUNK TEXT
# -----------------------------------
def chunk_text(text: str, chunk_size=1200):
    cleaned = clean(text)
    return [cleaned[i:i+chunk_size] for i in range(0, len(cleaned), chunk_size)]

# -----------------------------------
# SAFE STAGE-AWARE SUMMARIZATION
# -----------------------------------
def summarize_chunk(chunk: str, stage: str):
    if groq_client is None:
        return "Summary skipped (LLM not available)."

    try:
        system_prompt = STAGE_PROMPTS.get(stage, "Summarize this content.")

        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.2
        )
        return res.choices[0].message.content

    except Exception as e:
        print("[WARN] Summary failed:", e)
        return "Summary skipped."

# -----------------------------------
# BUILD VECTOR DB
# -----------------------------------
def build_vector_db(chunk_dict):
    os.makedirs(DB_DIR, exist_ok=True)
    db = PersistentClient(path=DB_DIR)

    try:
        db.delete_collection("crs")
    except:
        pass

    collection = db.create_collection(name="crs")

    count = 0
    for topic, chunks in chunk_dict.items():
        print(f"\n[DB] Adding topic → {topic} ({len(chunks)} chunks)")
        for idx, ck in enumerate(chunks):
            emb = embedder.encode(ck["text"]).tolist()
            collection.add(
                ids=[f"{topic}_{idx}"],
                embeddings=[emb],
                metadatas=[{
                    "topic": topic,
                    "summary": ck["summary"],
                    "stage": ck["stage"]
                }]
            )
            count += 1

    print(f"\n[DB] Stored {count} chunks.")

# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":
    print("\n===== CHUNKER v1.3 (SAFE MODE) =====\n")

    raw_data = load_raw_text()
    all_chunks = {}

    for topic, text in raw_data.items():
        print(f"\n[CHUNK] Processing → {topic}")
        parts = chunk_text(text)

        topic_chunks = []
        for part in parts:
            stage = detect_stage(part)
            summary = (
                "Content skipped (non-instructional text)."
                if is_junk_chunk(part)
                else summarize_chunk(part, stage)
            )

            topic_chunks.append({
                "text": part,
                "summary": summary,
                "stage": stage
            })

        all_chunks[topic] = topic_chunks

    build_vector_db(all_chunks)
    print("\n===== CHUNKER COMPLETE =====\n")
