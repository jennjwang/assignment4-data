"""
Write a script to filter language modeling data from a collection of Common Crawl WET files
(located under /shared-data/english-wet-data). You are free to apply any of the primitives
we’ve implemented in earlier parts of the assignment, and you’re also free to explore other
filters and methods for generating data (e.g., filtering based on n-gram language model
perplexity). Your goal is to produce data that, when trained on, minimizes the perplexity on
the C4 100 domains subset of the Paloma benchmark

Again, we note that you are allowed to make use of the Paloma validation data in
constructing filters or classifiers to process the CC WET files, but are not
allowed to literally copy any of the validation data into your training data.
Your script should report the number of examples kept by each filter that you’ve used, so you
have a sense of how the filters are contributing to the final output data.
"""

import concurrent.futures
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
from cs336_data.dedup import exact_dedup, minihash_dedup
import fasttext
import json

MODAL_DOMAINS_PATH = Path("/root/cs336_data/domains.txt")
DOMAINS_PATH = Path("cs336_data/domains.txt")

def load_allowed_domains(path: Path = DOMAINS_PATH):
    text = path.read_text()
    domains = set()
    for line in text.splitlines():
        if line.strip():
            domains.add( line.strip().lower())
    return domains

ALLOWED_DOMAINS = load_allowed_domains()
extract = TLDExtract()

# def in_domain(url: str, allowed: set[str] = ALLOWED_DOMAINS) -> bool:
#     result = extract(url)
#     host = result.registered_domain
#     if host in allowed:
#         return True
#     return False

# URL_RE = re.compile(
#     r"https?://[^\s<>\"')\]]+",
#     re.IGNORECASE,
# )

# def quality_filter(text):

#     if len(text) < 200:
#         return f"too short"
#         # return False
    
#     words = text.split()
#     word_count = len(words)

#     # Gopher filters
#     if word_count < 100 or word_count > 100000:
#         return f"word count"
#         # return False
        
#     mean_length = sum(len(w) for w in words) / word_count
#     if mean_length < 3 or mean_length > 10:
#         return f"word length"
#         # return False

#     lines = text.splitlines()
#     line_count = len(lines)
#     ellipsis_pcnt = sum([line.endswith("...") for line in lines]) / line_count
#     if ellipsis_pcnt > 0.3:
#         return f"ellipsis"
#         # return False

#     alpha_pcnt = sum(any(c.isalpha() for c in w) for w in words) / word_count
#     if alpha_pcnt < 0.8:
#         return f"alpha pnct"
#         # return False

#     mean_line_len = sum(len(l) for l in lines) / line_count
#     if mean_line_len < 50:
#         return "line length"

#     # # Ratio of unique words to total words above some threshold — catches repetitive/templated spam
#     # unique_word_count = len(set(words))
#     # if unique_word_count / word_count < 0.4:
#     #     return False
    
#     # Low URL density (few http:// occurrences per token) — removes link farms
#     url_count = len(URL_RE.findall(text))
#     url_pct = url_count / word_count
#     if url_pct > 0.3:
#         return f"url violation"
#         # return False
    
#     short_line_frac = sum([len(ln) < 30 for ln in lines]) / line_count
#     if short_line_frac > 0.4:
#         return "short lines"
    
#     if identify_language(text)[0] != 'en':
#         return 'not english'


#     # nav_frac = sum(ln.lstrip().startswith(('#','>','*','•')) for ln in lines) / line_count
#     # if nav_frac > 0.3:
#     #     return f"nav frac"
#     #     # return False
    
#     # if detect_NSFW(text)[0] == "nsfw" or detect_hatespeech(text)[0] == "toxic":
#     #     return "harmful"
#     #     return False

#     return True

def quality_filter(text):

    if len(text) < 200:
        # return f"too short"
        return False
    
    words = text.split()
    word_count = len(words)

    # Gopher filters
    if word_count < 100 or word_count > 100000:
        # return f"word count"
        return False
        
    mean_length = sum(len(w) for w in words) / word_count
    if mean_length < 3 or mean_length > 10:
        # return f"word length"
        return False

    lines = text.splitlines()
    line_count = len(lines)
    ellipsis_pcnt = sum([line.endswith("...") for line in lines]) / line_count
    if ellipsis_pcnt > 0.3:
        # return f"ellipsis"
        return False

    alpha_pcnt = sum(any(c.isalpha() for c in w) for w in words) / word_count
    if alpha_pcnt < 0.8:
        # return f"alpha pnct"
        return False

    # mean_line_len = sum(len(l) for l in lines) / line_count
    # if mean_line_len < 50:
    #     # return "line length"
    #     return False

    # short_line_frac = sum([len(ln) < 50 for ln in lines]) / line_count
    # if short_line_frac > 0.3:
    #     # return "short lines"
    #     return False
    
    lang, prob = identify_language(text)
    if not (lang == 'en'):
        # return 'not english'
        return False

    # nav_frac = sum(ln.lstrip().startswith(('#','>','*','•')) for ln in lines) / line_count
    # if nav_frac > 0.1:
    #     # return f"nav frac"
    #     return False
    
    # if detect_NSFW(text)[0] == "nsfw" or detect_hatespeech(text)[0] == "toxic":
    #     return "harmful"
    #     return False

    return True


def load_validation_documents(bin_path: str | Path | None = None) -> list[str]:
    if bin_path is None:
        bin_path = get_shared_assets_path() / "tokenized_paloma_c4_100_domains_validation.bin"

    data = np.fromfile(bin_path, dtype=np.uint16)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_token_id = tokenizer.eos_token_id

    eos_positions = np.where(data == eos_token_id)[0]
    documents: list[str] = []
    start = 0
    for pos in eos_positions:
        if pos > start:
            documents.append(tokenizer.decode(data[start:pos].tolist()))
        start = pos + 1
    if start < len(data):
        documents.append(tokenizer.decode(data[start:].tolist()))

    return documents

def _ft_line(label: str, text: str) -> str:
    text = " ".join(text.split())
    return f"__label__{label} {text}\n"

def sample_cc_negative(num: int, wet_dir: str = "/shared-data/english-wet-data"):
    out = []
    wet_paths = sorted(Path(wet_dir).glob("*.warc.wet.gz"))
    random.shuffle(wet_paths)
    for path in wet_paths:
        if len(out) >= num: break
        with open(path, "rb") as f:
        #     for record in ArchiveIterator(f, parse_http=True):
        #         if record.record_type != WarcRecordType.response:
        #             continue
        #         html_content = record.reader.read()
        #         text = extract_text_from_html_bytes(html_content)
            for record in ArchiveIterator(f):       
                if record.record_type != WarcRecordType.conversion:
                    continue
                if len(out) >= num: break
                text = record.reader.read().decode("utf-8", errors="replace")
                if not text:
                    continue
                if not quality_filter(text):
                    continue
                out.append(text.replace("\n", " "))
                if len(out) >= num:
                    break
    return out

def split_list(items, train_frac):
    xs = list(items)
    random.Random(42).shuffle(xs)
    n = len(xs)
    n_train = int(train_frac * n)
    if n >= 2:
        n_train = min(max(n_train, 1), n-1)
    else:
        n_train = 1
    return xs[:n_train], xs[n_train:]

def train_classifier(
    train_frac: float = 0.9,
    wet_dir: str = "/shared-data/english-wet-data" ):

    out_dir = Path("/root/data/classifiers")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "paloma_train.txt"
    test_path = out_dir / "paloma_test.txt"
    model_path = out_dir / "paloma_model.bin"

    paloma_docs = load_validation_documents()
    paloma_docs = paloma_docs
    cc_docs = sample_cc_negative(len(paloma_docs), wet_dir=wet_dir)

    (out_dir / "paloma_docs.json").write_text(json.dumps(paloma_docs, ensure_ascii=False))
    (out_dir / "cc_docs.json").write_text(json.dumps(cc_docs, ensure_ascii=False))
    print(f"Saved {len(paloma_docs)} paloma docs and {len(cc_docs)} cc docs to {out_dir}", flush=True)

    train_p, test_p = split_list(paloma_docs, train_frac)
    train_c, test_c = split_list(cc_docs, train_frac)
    
    with train_path.open("w", encoding="utf-8") as f:
        for doc in train_p:
            f.write(_ft_line("paloma", doc))
        for doc in train_c:
            f.write(_ft_line("cc", doc))

    with test_path.open("w", encoding="utf-8") as f:
        for doc in test_p:
            f.write(_ft_line("paloma", doc))
        for doc in test_c:
            f.write(_ft_line("cc", doc))

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=25,
        lr=1.0,
        wordNgrams=2,
        dim=64,
    )

    n_ex, precision, recall = model.test(str(test_path))
    print(f"test: n={n_ex} P={precision:.3f} R={recall:.3f}")

    model.save_model(str(model_path))

# def process_single_wet_file(nput_path: str, tier1_path: str, tier2_path: str,) -> dict[str, int]:
#     """Filter one WET file; route quality-pass docs to tier1 (in-domain) or tier2."""
#     counts: dict[str, int] = {"total": 0, "tier1": 0, "tier2": 0}
#     with (
#         open(input_path, "rb") as f,
#         open(tier1_path, "w") as tier1_out,
#         open(tier2_path, "w") as tier2_out,
#     ):
#         for record in ArchiveIterator(f):
#             if record.record_type != WarcRecordType.conversion:
#                 continue
#             counts["total"] += 1
#             url = record.headers.get("WARC-Target-URI", "")
#             text = record.reader.read().decode("utf-8", errors="replace")

#             quality_res = quality_filter(text)
#             if quality_res is not True:
#                 key = quality_res
#                 counts[key] = counts.get(key, 0) + 1
#                 continue

#             if in_domain(url):
#                 tier1_out.write(text)
#                 tier1_out.write("\n")
#                 counts["tier1"] += 1
#             else:
#                 tier2_out.write(text)
#                 tier2_out.write("\n")
#                 counts["tier2"] += 1
#     return counts


# def summarize_validation_filters(documents: list[str] | None = None) -> dict[str, int]:
#     if documents is None:
#         documents = load_validation_documents()

#     counts: dict[str, int] = {"pass": 0}
#     for text in documents:
#         result = quality_filter(text)
#         if result is True:
#             counts["pass"] += 1
#         else:
#             reason = result if isinstance(result, str) else "too_short"
#             counts[reason] = counts.get(reason, 0) + 1
#     return counts


# if __name__ == "__main__":
#     # counts = summarize_validation_filters()
#     # total = sum(counts.values())
#     # print(f"Analyzed {total} validation documents")
#     # for reason, n in sorted(counts.items(), key=lambda x: -x[1]):
#     #     print(f"  {reason}: {n} ({100 * n / total:.1f}%)")
#     train_classifier()

MODAL_WET_DIR = Path("/shared-data/english-wet-data")
# MODAL_OUTPUT_DIR = Path("/root/data/filtered-wet-text")
# MODAL_TIER1_DIR = MODAL_OUTPUT_DIR / "tier1"
# MODAL_TIER2_DIR = MODAL_OUTPUT_DIR / "tier2"


# def _merge_counts(total: dict[str, int], counts: dict[str, int]) -> dict[str, int]:
#     for key, value in counts.items():
#         total[key] = total.get(key, 0) + value
#     return total


# @app.function(
#     image=build_image(),
#     volumes=VOLUME_MOUNTS,
#     timeout=60 * 60 * 4,
#     cpu=1,
# )
# def filter_wet_file_modal(wet_path: str) -> dict[str, int]:
#     MODAL_TIER1_DIR.mkdir(parents=True, exist_ok=True)
#     MODAL_TIER2_DIR.mkdir(parents=True, exist_ok=True)
#     name = Path(wet_path).name
#     tier1_path = MODAL_TIER1_DIR / f"{name}.txt"
#     tier2_path = MODAL_TIER2_DIR / f"{name}.txt"
#     return process_single_wet_file(wet_path, str(tier1_path), str(tier2_path))


# @app.function(
#     image=build_image(),
#     volumes=VOLUME_MOUNTS,
#     timeout=60 * 60 * 12,
# )
# def run_filtering(max_files: int | None = None) -> dict[str, int]:
#     wet_paths = sorted(MODAL_WET_DIR.glob("*.warc.wet.gz"))
#     if max_files is not None:
#         wet_paths = wet_paths[:max_files]

#     print(f"Filtering {len(wet_paths)} WET files -> {MODAL_OUTPUT_DIR}", flush=True)
#     total_counts = {}
#     for counts in filter_wet_file_modal.map([str(p) for p in wet_paths]):
#         _merge_counts(total_counts, counts)
#         print(counts, flush=True)

#     data_volume.commit()
#     print(f"Done. Aggregate counts: {total_counts}", flush=True)
#     return total_counts


# @app.local_entrypoint()
# def main(max_files: int = 0) -> None:
#     limit = max_files if max_files > 0 else None
#     run_filtering.remote(max_files=limit)


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 2,
)
def run_train_classifier(train_frac: float = 0.9) -> None:
    train_classifier(
        train_frac=train_frac,
        wet_dir=str(MODAL_WET_DIR)
    )
    data_volume.commit()


@app.local_entrypoint()
def train() -> None:
    run_train_classifier.remote()
