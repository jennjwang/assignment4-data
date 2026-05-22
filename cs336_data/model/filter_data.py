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

MODAL_WET_DIR = Path("/shared-data/english-wet-data")
MODAL_OUT_DIR = Path("/root/data/filtered-wet-text")

MODEL_PATH = Path("/root/data/classifiers/paloma_model.bin")

# at module scope (loads once per worker process)
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = fasttext.load_model(str(MODEL_PATH))
    return _MODEL

def cut_short_lines(text):
    lines = text.splitlines()
    new_lines = []
    for l in lines:
        if len(l.strip()) < 30:
            continue
        new_lines.append(l)
    return "\n".join(new_lines)

def quality_filter(text):

    if len(text) < 200:
        return f"too short"
        # return False
    
    words = text.split()
    word_count = len(words)

    # Gopher filters
    if word_count < 100 or word_count > 100000:
        return f"word count"
        # return False
        
    mean_length = sum(len(w) for w in words) / word_count
    if mean_length < 3 or mean_length > 10:
        return f"word length"
        # return False

    lines = text.splitlines()
    line_count = len(lines)
    ellipsis_pcnt = sum([line.endswith("...") for line in lines]) / line_count
    if ellipsis_pcnt > 0.3:
        return f"ellipsis"
        # return False

    alpha_pcnt = sum(any(c.isalpha() for c in w) for w in words) / word_count
    if alpha_pcnt < 0.8:
        return f"alpha pnct"
        return False

    # mean_line_len = sum(len(l) for l in lines) / line_count
    # if mean_line_len < 50:
    #     # return "line length"
    #     return False

    # short_line_frac = sum([len(ln) < 50 for ln in lines]) / line_count
    # if short_line_frac > 0.3:
    #     return "short lines"
    #     # return False
    
    lang, prob = identify_language(text)
    if not (lang == 'en'):
        return 'not english'
        # return False

    # nav_frac = sum(ln.lstrip().startswith(('#','>','*','•')) for ln in lines) / line_count
    # if nav_frac > 0.1:
    #     # return f"nav frac"
    #     return False
    
    # if detect_NSFW(text)[0] == "nsfw" or detect_hatespeech(text)[0] == "toxic":
    #     return "harmful"
    #     return False

    return True


def classify_quality(text: str) -> tuple[str, float]:
    # path = get_shared_assets_path() / "classifiers" / "quality_model.bin"
    # path = "local-shared-data/classifiers/quality_model.bin"
    # model = fasttext.load_model(str(MODEL_PATH))
    model=  _get_model()

    text = text.replace("\n", " ")
    labels, probs = model.predict(text)
    label = labels[0].removeprefix("__label__")
    return label, float(probs[0])

def process_single_wet_file(input_path: str, output_path: str,) -> dict[str, int]:
    counts = {"total": 0, "output": 0}
    with (
        open(input_path, "rb") as f,
        open(output_path, "w") as out,
    ):
        for record in ArchiveIterator(f):
            if record.record_type != WarcRecordType.conversion:
                continue
            counts["total"] += 1
            # url = record.headers.get("WARC-Target-URI", "")
            text = record.reader.read().decode("utf-8", errors="replace")

            res = quality_filter(text)
            if res is not True:
                counts[res] = counts.get(res, 0) + 1
                continue

            # quality_res = quality_filter(text)
            label, prob = classify_quality(text)
            if (label != "paloma" and prob > 0.6):
                key = "classifier rejection"
                counts[key] = counts.get(key, 0) + 1
                continue

            text = cut_short_lines(text)
            out.write(text)
            out.write("\n<|endoftext|>\n")
            counts["output"] += 1
    return counts

def _merge_counts(total: dict[str, int], counts: dict[str, int]) -> dict[str, int]:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value
    return total



@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
    cpu=1,
)
def filter_wet_file_modal(wet_path: str) -> dict[str, int]:
    MODAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(wet_path).name
    out_path = MODAL_OUT_DIR / f"{name}.txt"
    return process_single_wet_file(wet_path, str(out_path))

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 12,
)
def run_filtering(max_files: int | None = None) -> dict[str, int]:
    # a = list(Path("/shared-data/english-wet-data").iterdir())
    # print(len(a))
    # return

    wet_paths = sorted(MODAL_WET_DIR.glob("*.warc.wet.gz"))
    
    if max_files is not None:
        wet_paths = wet_paths[:max_files]

    # model = fasttext.load_model(str(path))

    print(f"Filtering {len(wet_paths)} WET records -> {MODAL_OUT_DIR}", flush=True)
    total_counts = {}
    for counts in filter_wet_file_modal.map([str(p) for p in wet_paths]):
        _merge_counts(total_counts, counts)
        print(counts, flush=True)

    data_volume.commit()
    print(f"Done. Aggregate counts: {total_counts}", flush=True)

    # Done. Aggregate counts: {'total': 15923174, 'output': 192993, 'classifier rejection': 13311541, 'word count': 698452, 'too short': 1005724, 'alpha pnct': 698718, 'ellipsis': 7014, 'word length': 8732}
    # Done. Aggregate counts: {'total': 15923174, 'output': 1382848, 'classifier rejection': 14540326}
    # Done. Aggregate counts: {'total': 15923174, 'output': 1342400, 'classifier rejection': 13827727, 'alpha pnct': 753047}
    # Aggregate counts: {'total': 15923174, 'output': 210364, 'not english': 12604935, 'classifier rejection': 2354828, 'alpha pnct': 753047}
    # Done. Aggregate counts: {'total': 15923174, 'output': 954009, 'classifier rejection': 7412469, 'alpha pnct': 753047, 'not english': 6803649}
    # Done. Aggregate counts: {'total': 15923174, 'output': 245800, 'classifier rejection': 7261482, 'word count': 698452, 'too short': 1005724, 'alpha pnct': 698718, 'not english': 5997252, 'ellipsis': 7014, 'word length': 8732}
    return total_counts

@app.local_entrypoint()
def main(max_files: int = 0) -> None:
    limit = max_files if max_files > 0 else None
    run_filtering.remote(max_files=limit)
