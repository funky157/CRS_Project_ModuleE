# =====================================================
# CRS PROJECT
# File: chunker.py
# Version: v1.1
# Status: Stable (Optimized)
# Purpose: Chunk, summarize (limited), embed, and store
# Author: Harish
# Last Updated: 2025-01-XX
# =====================================================

import os
import glob
import re
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -----------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------
load_dotenv("../.env")

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("Missing GROQ_API_KEY in .env")

groq_client = Groq(api_key=GROQ_KEY)

# Local embedding model (FREE, offline)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

RAW_DIR = "db/raw/"
DB_DIR = "db/chroma/"

# -----------------------------------------------------
# CONFIG (THIS IS WHAT MAKES v1.1 BETTER)
# -----------------------------------------------------
CHUNK_SIZE = 2500          # Increased from 1200
MAX_SUMMARIES_PER_TOPIC = 10   # Hard limit


# -----------------------------------------------------
# UTILS
# -----------------------------------------------------
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------
# LOAD RAW DATA
# -----------------------------------------------------
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
                print(f"[WARN] Skipped empty file: {f}")

    print(f"[LOAD] Loaded {len(data)} valid topics")
    return data


# -----------------------------------------------------
# CHUNKING
# -----------------------------------------------------
def chunk_text(text: str):
    cleaned = clean(text)
    chunks = []

    for i in range(0, len(cleaned), CHUNK_SIZE):
        chunks.append(cleaned[i:i + CHUNK_SIZE])

    return chunks


# -----------------------------------------------------
# SUMMARIZATION (LIMITED)
# -----------------------------------------------------
def summarize_chunk(chunk: str):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this electronics content in 4–5 simple lines for quick revision."
                },
                {"role": "user", "content": chunk}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Summary failed: {e}")
        return "Summary unavailable."


# -----------------------------------------------------
# EMBEDDING (LOCAL)
# -----------------------------------------------------
def embed(text: str):
    try:
        return embedder.encode(text).tolist()
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None


# -----------------------------------------------------
# BUILD VECTOR DATABASE
# -----------------------------------------------------
def build_vector_db(chunk_dict):
    os.makedirs(DB_DIR, exist_ok=True)
    db = PersistentClient(path=DB_DIR)

    try:
        db.delete_collection("crs")
    except:
        pass

    collection = db.create_collection(name="crs")
    total = 0

    for topic, chunks in chunk_dict.items():
        print(f"\n[DB] Indexing topic → {topic}")

        for idx, ck in enumerate(chunks):
            emb = embed(ck["text"])
            if emb is None:
                continue

            collection.add(
                ids=[f"{topic}_{idx}"],
                embeddings=[emb],
                documents=[ck["text"]],
                metadatas=[{
                    "topic": topic,
                    "summary": ck["summary"]
                }]
            )
            total += 1

    print(f"\n[DB] Vector DB ready. Total chunks indexed: {total}")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    print("\n===== CHUNKER v1.1 STARTED =====\n")

    raw_data = load_raw_text()
    final_chunks = {}

    for topic, text in raw_data.items():
        print(f"\n[CHUNK] Processing topic: {topic}")
        parts = chunk_text(text)

        topic_chunks = []
        for i, part in enumerate(parts):
            # Only summarize first N chunks
            if i < MAX_SUMMARIES_PER_TOPIC:
                summary = summarize_chunk(part)
            else:
                summary = "Summary skipped (beyond limit)."

            topic_chunks.append({
                "text": part,
                "summary": summary
            })

        final_chunks[topic] = topic_chunks

    build_vector_db(final_chunks)

    print("\n===== CHUNKER v1.1 COMPLETE =====\n")
