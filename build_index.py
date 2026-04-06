"""Pre-build embedding index locally (GPU), then upload cache files to server.

Usage:
    python build_index.py

This creates _cache_embeddings.npy and _cache_chunks.json in each collection dir.
Upload data/collections/ to the server — it will skip re-indexing on startup.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import config
from indexer import IndexedCollection

def main():
    for role in ("optimist", "pessimist"):
        col_dir = os.path.join(config.COLLECTIONS_DIR, role)
        if not os.path.isdir(col_dir):
            print(f"  {role}: directory not found, skipping")
            continue

        txt_files = [f for f in os.listdir(col_dir) if f.endswith(".txt")]
        if not txt_files:
            print(f"  {role}: no .txt files, skipping")
            continue

        print(f"Building index for '{role}' ({len(txt_files)} files)...")
        t0 = time.time()
        col = IndexedCollection.build(
            name=role,
            directory=col_dir,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            model_name=config.EMBEDDING_MODEL_NAME,
        )
        dt = time.time() - t0
        print(f"  {len(col.chunks)} chunks indexed in {dt:.1f}s")
        print(f"  Cache saved to {col_dir}/_cache_*")
        print()

    print("Done! Upload data/collections/ to the server.")


if __name__ == "__main__":
    main()
