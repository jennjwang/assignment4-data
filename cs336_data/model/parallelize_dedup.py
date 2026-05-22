from cs336_data.dedup import exact_dedup, minihash_dedup
import concurrent.futures
import functools
import os
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
from cs336_data.common import get_shared_assets_path
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_emails, mask_phones, mask_ips
from cs336_data.harmful_content import detect_NSFW, detect_hatespeech
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes
import random
import modal
from pathlib import Path
from nltk.tokenize import word_tokenize
import re
from tldextract import TLDExtract
from cs336_data.modal_utils import app, build_image, VOLUME_MOUNTS, data_volume
import numpy as np
from transformers import AutoTokenizer
from cs336_data.dedup import exact_dedup, minihash_dedup, minhash, lsh, normalize
import fasttext
import json
from collections import defaultdict
import pickle

MODAL_OUT_DIR = Path("/root/data/filtered-wet-text")
EXACT_DEDUP_DIR = Path("/root/data/deduped/exact-filtered-wet-text")
MINHASH_DEDUP_DIR = Path("/root/data/deduped/minhash-filtered-wet-text")
EOS = "\n<|endoftext|>\n"

def exact_doc_dedup(inputs, output) -> dict:
    hash_freq = {}
    for p in inputs:
        text = Path(p).read_text(encoding="utf-8")
        for doc in text.split(EOS):
            doc = doc.strip()
            if not doc:
                continue
            hash_freq[hash(doc)] = hash_freq.get(hash(doc), 0) + 1

    total_in = total_out = 0
    for p in inputs:
        text = Path(p).read_text(encoding="utf-8")
        kept = []
        for doc in text.split(EOS):
            doc = doc.strip()
            if not doc:
                continue
            total_in += 1
            if hash_freq[hash(doc)] == 1:
                kept.append(doc)
                total_out += 1

        output_path = Path(output) / Path(p).name
        output_path.write_text(EOS.join(kept) + EOS if kept else "", encoding="utf-8")

    return {"total_in": total_in, "total_out": total_out, "removed": total_in - total_out}
        

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
)
def run_exact_dedup():
    inputs = sorted(MODAL_OUT_DIR.glob("*.txt"))
    exact_dedup_dir = EXACT_DEDUP_DIR
    exact_dedup_dir.mkdir(parents=True, exist_ok=True)
    print(exact_doc_dedup(inputs, exact_dedup_dir))
    # {'total_in': 1382848, 'total_out': 487624, 'removed': 895224}
    # {'total_in': 105753, 'total_out': 42544, 'removed': 63209}
    # {'total_in': 954009, 'total_out': 321573, 'removed': 632436}
    # {'total_in': 189249, 'total_out': 176838, 'removed': 12411}
    data_volume.commit()

@app.local_entrypoint()
def modal_run_exact_dedup():
    run_exact_dedup.remote()
    # count_dedup.remote()

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
)
def count_docs_in_file(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    docs = [d for d in text.split("\n<|endoftext|>\n") if d.strip()]
    return len(docs)

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
)
def count_dedup(dir_path):
    paths = sorted(dir_path.glob("*.txt"))
    total = sum(count_docs_in_file.map([str(p) for p in paths]))
    print(f"Total docs after dedup: {total} across {len(paths)} files")
    return total


SIGNATURES_DIR = Path("/root/data/deduped/minhash-signatures")
N_GRAM_LEN = 5
NUM_HASHES = 100
NUM_BANDS = 10
THRESHOLD = 0.7

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4
)
def compute_sigs_for_file(path):
    text = Path(path).read_text(encoding="utf-8")
    docs = text.split("\n<|endoftext|>\n")
    out_dir = SIGNATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sigs = []
    doc_ids = []
    ngram_sets = {}  # j -> set(n_grams), keyed by enumerate index
    for j, doc in enumerate(docs):
        doc = doc.strip()
        if not doc:
            continue
        words = normalize(doc)
        n_grams = [tuple(words[i: (i + N_GRAM_LEN)]) for i in range(len(words) - N_GRAM_LEN + 1)]
        if len(n_grams) == 0:
            continue
        signature = minhash(n_grams, NUM_HASHES)
        ngram_sets[j] = set(n_grams)
        sigs.append(signature)
        doc_ids.append(f"{Path(path).stem}:{j}")
    
    sigs_array = np.array(sigs, dtype=np.uint64) 
    out_stem = Path(out_dir) / f"{Path(path).stem}"
    np.save(f"{out_stem}.npy", sigs_array)
    with open(f"{out_stem}.ngrams.pkl", "wb") as f:
        pickle.dump(ngram_sets, f)
    json.dump(doc_ids, open(f"{out_stem}.json", "w"))
    data_volume.commit()
    return str(out_stem) + ".npy"

    # doc_to_hash[p] = minhashes
    # doc_to_ngrams[p] = set(n_grams)

def find_candidate_pairs_lsh():
    doc_to_hash = {}
    for npy_path in sorted(SIGNATURES_DIR.glob("*.npy")):
        json_path = npy_path.with_suffix(".json")
        sigs = np.load(npy_path)          # shape (n_docs, K)
        doc_ids = json.loads(json_path.read_text())
        for doc_id, sig in zip(doc_ids, sigs):
            doc_to_hash[doc_id] = sig

    pairs = lsh(num_bands=NUM_BANDS, num_hashes=NUM_HASHES, doc_to_hash=doc_to_hash)
    return list(pairs)

@functools.lru_cache(maxsize=128)
def _get_ngrams(file_stem):
    path = SIGNATURES_DIR / f"{file_stem}.ngrams.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

VERIFY_BATCH_SIZE = 1000

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
    memory=8192,
)
def verify_pairs_batch(pair_list):
    verified = []
    for pair in pair_list:
        doc_id_a, doc_id_b = pair
        file_a, idx_a = doc_id_a.rsplit(":", 1)
        file_b, idx_b = doc_id_b.rsplit(":", 1)
        grams_a = _get_ngrams(file_a)[int(idx_a)]
        grams_b = _get_ngrams(file_b)[int(idx_b)]
        if not grams_a or not grams_b:
            # return None
            continue
        jaccard = len(grams_a & grams_b) / len(grams_a | grams_b)
        if jaccard > THRESHOLD:
            verified.append((doc_id_a, doc_id_b, jaccard))
            # return (doc_id_a, doc_id_b, jaccard)
        # return None
    return verified

def cluster_and_remove(verified_pairs: list[tuple[str, str, float]]) -> set[str]:
    parent = {} 
    def find(x):  
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(x, y):
        # merge x's and y's clusters by joining their roots
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx 

    for a, b, _ in verified_pairs:
        union(a, b)

    clusters = defaultdict(set)
    for doc in parent:
        clusters[find(doc)].add(doc)
    
    to_remove = set()
    for cluster in clusters.values():
        keeper = random.choice(list(cluster))
        to_remove.update(cluster - {keeper})
    
    return to_remove

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4
)
def write_output_doc(args):
    path, to_remove = args
    text = Path(path).read_text(encoding="utf-8")
    docs = text.split("\n<|endoftext|>\n")
    file_stem = Path(path).stem
    
    remaining = []
    for j, doc in enumerate(docs):
        doc = doc.strip()
        if not doc:
            continue
        doc_id = f"{file_stem}:{j}"
        if doc_id not in to_remove:
            remaining.append(doc)
    
    out_path = MINHASH_DEDUP_DIR / Path(path).name
    out_path.write_text("\n<|endoftext|>\n".join(remaining), encoding="utf-8")
    return str(out_path)


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
    memory=16384,
)
def run_parallel_dedup():
    import shutil
    if SIGNATURES_DIR.exists():
        shutil.rmtree(SIGNATURES_DIR)
    SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)

    inputs = sorted(EXACT_DEDUP_DIR.glob("*.txt"))
    input_strs = [str(p) for p in inputs]
    sig_paths = list(compute_sigs_for_file.map(input_strs))
    data_volume.reload()
    MINHASH_DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    candidate_pairs = find_candidate_pairs_lsh()
    batches = [
        candidate_pairs[i: i + VERIFY_BATCH_SIZE]
        for i in range(0, len(candidate_pairs), VERIFY_BATCH_SIZE)
    ]
    verified = []
    for batch_results in verify_pairs_batch.map(batches):
        verified.extend(batch_results)
    
    to_remove = cluster_and_remove(verified)
    args = [(p, to_remove) for p in input_strs]
    for _ in write_output_doc.map(args):
        pass
    data_volume.commit()
    return len(to_remove)

@app.local_entrypoint()
def minhash_dedup():
    # Total docs after exact dedup: 487624 across 625 files
    # Total docs after minhash dedup: 459926 across 625 files
    # Total docs after minhash dedup: 307734 across 625 files
    # Total docs after minhash dedup: 176389 across 625 files
    run_parallel_dedup.remote()
    count_dedup.remote(MINHASH_DEDUP_DIR)
