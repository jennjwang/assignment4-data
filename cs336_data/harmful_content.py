"""
Deliverable: A function that labels a given string as containing NSFW content or not,
returning a pair containing both the label and a confidence score. Implement the adapter
[run_classify_nsfw] and make sure it passes uv run pytest -k test_classify_nsfw. Note
that this test is just a sanity check, taken from the Jigsaw dataset, but by no means asserts
that your classifier is accurate, which you should validate.
"""

from nltk.tokenize import word_tokenize
import fasttext
from cs336_data.common import get_shared_assets_path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes
import random

def detect_NSFW(text):
    p = get_shared_assets_path() / "classifiers" / "dolma_fasttext_nsfw_jigsaw_model.bin"
    
    model = fasttext.load_model(str(p))

    text = text.replace("\n", "")
    labels, probabilities = model.predict(text)

    label = labels[0]
    code = label.removeprefix("__label__")

    return code, probabilities[0]

def detect_hatespeech(text):
    p = get_shared_assets_path() / "classifiers" / "dolma_fasttext_hatespeech_jigsaw_model.bin"
    
    model = fasttext.load_model(str(p))

    text = text.replace("\n", "")
    labels, probabilities = model.predict(text)

    label = labels[0]
    code = label.removeprefix("__label__")

    return code, probabilities[0]


if __name__ == "__main__":
    
    candidates = []
    with open("local-shared-data/CC/example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, parse_http=True):
            if record.record_type == WarcRecordType.response:
                record_id=record.headers["WARC-Record-ID"]
                html_content = record.reader.read()
                text = extract_text_from_html_bytes(html_content)
                candidates.append((record_id, text))
        
    random.seed(42)
    sample = random.sample(candidates, k=min(20, len(candidates)))
    for record_id, text in sample:
        label, score = detect_NSFW(text)
        with open(f"results/nsfw/{label}_example_{record_id}.txt", "w") as f:
            f.write(text)
        
        label, score = detect_hatespeech(text)
        with open(f"results/hatespeech/{label}_example_{record_id}.txt", "w") as f:
            f.write(text)
