#!/usr/bin/env python3
"""
query_ollama_clean.py

Usage:
    python query_ollama_clean.py
Then enter the file path when prompted, then use the interactive SEARCH> prompt.
"""

import os
# suppress TF logs/warnings early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 0=all, 1=info, 2=warning, 3=error
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import json
import logging
import warnings
from pathlib import Path
from time import time

# reduce noisy package warnings (protobuf)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Logging: pretty compact CLI output + json info log to file
logger = logging.getLogger("__main__")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%dT%H:%M:%S")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

# also store structured events to a log file
fh = logging.FileHandler("query_ollama_clean.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(message)s'))  # we'll log JSON strings
logger.addHandler(fh)


def log_event(event: dict):
    # write short JSON line to file handler and info to console
    j = json.dumps(event, default=str)
    fh_stream = logger.handlers[1]  # file handler
    fh_stream.emit(logging.LogRecord(name="__main__", level=logging.INFO, pathname=__file__,
                                     lineno=0, msg=j, args=(), exc_info=None))
    # also print a compact info line to console
    logger.info(event.get("message", event.get("event", json.dumps(event))))


# file reading helpers
def read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)


def read_pdf(path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join([t.strip() for t in texts if t and t.strip()])


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    ext = p.suffix.lower()
    if ext == ".docx":
        return read_docx(str(p))
    if ext == ".pdf":
        return read_pdf(str(p))
    if ext in (".txt", ".md"):
        return read_txt(str(p))
    raise ValueError("Unsupported file. Supported extensions: .docx, .pdf, .txt, .md")


# simple chunker: split into chunks up to max_chars preserving words
def chunk_text(text: str, max_chars: int = 800):
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w) + 1
        else:
            cur.append(w)
            cur_len += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# embeddings + FAISS
import numpy as np
import faiss


def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=32):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embs


def build_faiss_index(embeddings: np.ndarray):
    # Use IndexFlatIP (inner product) and normalize vectors -> cosine similarity
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    # normalize: avoid divide by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = embeddings / norms
    index.add(embeddings_norm.astype(np.float32))
    return index


def search_index(index, embeddings, texts, query_vec, top_k=5):
    # normalize query_vec
    qn = query_vec.astype(np.float32)
    qn = qn / (np.linalg.norm(qn) + 1e-12)
    scores, idxs = index.search(qn.reshape(1, -1), top_k)
    hits = []
    for sc, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        hits.append({"score": float(sc), "idx": int(idx), "text": texts[idx]})
    return hits


# Ollama integration: support streaming JSON-lines responses and collect 'response' fields
import requests


def ollama_generate_collect(url: str, model: str, prompt: str, timeout=60):
    endpoint = url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "max_tokens": 512}
    headers = {"Content-Type": "application/json"}
    try:
        # stream=True to catch chunked JSON lines
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout, stream=True)
    except requests.RequestException as e:
        return False, f"Ollama request failed: {e}"

    if resp.status_code != 200:
        try:
            return False, f"Ollama returned {resp.status_code}: {resp.text}"
        except Exception:
            return False, f"Ollama returned {resp.status_code}"

    collected = []
    # The Ollama server often streams JSON objects line-by-line
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            # Some servers prefix with non-json characters, try to parse carefully
            line = raw.strip()
            try:
                obj = json.loads(line)
            except Exception:
                # If it's not a pure JSON object, try to find a JSON substring
                try:
                    start = line.find("{")
                    obj = json.loads(line[start:]) if start != -1 else {"raw": line}
                except Exception:
                    obj = {"raw": line}
            # the streaming shape often includes {"response":"...","done":false}
            if isinstance(obj, dict) and "response" in obj:
                collected.append(str(obj.get("response") or ""))
            elif isinstance(obj, dict) and "text" in obj:
                collected.append(str(obj.get("text") or ""))
            else:
                # ignore other meta chunks
                continue
    except requests.RequestException as e:
        return False, f"Ollama streaming error: {e}"

    answer = "".join(collected).strip()
    return True, answer


def pretty_snippet(text, max_len=300):
    s = text.strip().replace("\n", " ")
    return (s[:max_len].rstrip() + "...") if len(s) > max_len else s


def main():
    # check Ollama reachable early (non-fatal)
    default_ollama_url = "http://127.0.0.1:11434"
    default_model = "llama3:latest"

    # prompt file path
    try:
        file_path = input("Enter file path: ").strip()
        if not file_path:
            print("No file given. Exiting.")
            return
    except KeyboardInterrupt:
        print("\nExit.")
        return

    # log start event
    log_event({"event": "app.start", "message": f"file={file_path}", "model": default_model, "use_embed": True})

    # read file
    try:
        t0 = time()
        text = read_file(file_path)
        log_event({"event": "read_file.ok", "path": file_path, "chars": len(text), "duration_s": time() - t0})
    except Exception as e:
        log_event({"event": "read_file.error", "err": str(e)})
        print(f"Error reading file: {e}")
        return

    if not text.strip():
        print("File contains no text.")
        return

    print("[STEP] Chunking...")
    chunks = chunk_text(text, max_chars=800)
    log_event({"event": "chunks.created", "num_chunks": len(chunks)})

    # compute embeddings
    try:
        print("[STEP] Computing embeddings (this may take a little)...")
        t0 = time()
        embs = embed_texts(chunks, model_name="all-MiniLM-L6-v2")
        log_event({"event": "build_index.embedded", "secs": time() - t0, "dim": embs.shape[1], "chunks": len(chunks)})
    except Exception as e:
        log_event({"event": "embed.error", "err": str(e)})
        print("Embedding failed:", e)
        return

    # build FAISS index (always use IndexFlatIP + normalized vectors)
    index = build_faiss_index(embs)
    log_event({"event": "faiss.built", "n": index.ntotal})

    # interactive search loop
    print("Ready. Empty query to exit.")
    while True:
        try:
            q = input("SEARCH> ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break
        if not q:
            print("Bye.")
            break

        # compute query embedding
        try:
            q_emb = embed_texts([q], model_name="all-MiniLM-L6-v2")
        except Exception as e:
            print("Query embedding failed:", e)
            continue

        top_k = 5
        hits = search_index(index, embs, chunks, q_emb[0], top_k=top_k)
        log_event({"event": "search.done", "query": q, "hits": len(hits)})

        print("\n--- TOP HITS ---")
        for i, h in enumerate(hits, 1):
            print(f"{i}. score={h['score']:.4f}")
            print("Snippet:", pretty_snippet(h["text"], max_len=300))
            print("-" * 40)
        if not hits:
            print("No results found.")
            continue

        # prepare prompt for Ollama: include top contexts (concise)
        contexts = "\n\n---\n\n".join([f"[context {i+1}]\n{h['text']}" for i, h in enumerate(hits)])
        prompt = f"You are a helpful assistant. Use the following contexts to answer the user's question.\n\nCONTEXTS:\n{contexts}\n\nQUESTION:\n{q}\n\nProvide a concise answer and mention which context index you used (e.g., context 1)."
        print("\n[STEP] Calling Ollama...")
        log_event({"event": "ollama.call.start", "model": default_model, "endpoint": default_ollama_url + "/api/generate"})

        ok, ans = ollama_generate_collect(default_ollama_url, default_model, prompt)
        if not ok:
            log_event({"event": "ollama.call_failed", "err": ans})
            print("\n[Ollama error] ", ans)
            print("\nTip: Ensure `ollama serve` is running and model name is correct (use `ollama ls`).")
            continue

        # print clean answer
        print("\n--- OLLAMA ANSWER ---\n")
        print(ans.strip() + "\n")
        log_event({"event": "ollama.call.ok", "answer_excerpt": ans.strip()[:200]})

    # exit loop
    return


if __name__ == "__main__":
    main()
