"""
a) Train a quality classifier that, given text, returns a numeric quality score.
Deliverable: A quality classifier for use in the next subproblem.
(b) Write a function that labels a page as high or low-quality, and provides a confidence score in
the label.
Deliverable: A function taking a string as its only argument, and returning a pair with a
label (high-quality or not) and a confidence score. Implement the adapter
[run_classify_quality] . As a sanity check, make sure it correctly classifies the two examples
we provide by running uv run pytest -k test_classify_quality
"""

import fasttext
from cs336_data.common import get_shared_assets_path
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_emails, mask_phones, mask_ips
from cs336_data.harmful_content import detect_NSFW, detect_hatespeech
from cs336_data.quality_rules import gopher_filters
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes
import random
import modal
from pathlib import Path
from cs336_data.modal_utils import app, build_image, VOLUME_MOUNTS, data_volume

'''
TRAIN_PATH = Path("/root/data/quality_training/train.txt")

def process_warc(path, label):
    n = 0
    with open(TRAIN_PATH, 'a') as train_out:
        with open(path, 'rb') as f:
            for record in ArchiveIterator(f, parse_http=True):
                if record.record_type == WarcRecordType.response:
                    record_id=record.headers["WARC-Record-ID"]
                    html_content = record.reader.read()
                    text = extract_text_from_html_bytes(html_content)
                    if not text:
                        continue
                    if identify_language(text)[0] != "en":
                        continue
                    if detect_NSFW(text)[0] == "nsfw" or detect_hatespeech(text)[0] == "toxic":
                        continue
                    if not gopher_filters(text):
                        continue
                    
                    line = f"__label__{label} {text.replace(chr(10), ' ')}\n"
                    train_out.write(line)
                    n += 1
    return n

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 4,
)
def process_warc_modal(warc_path: str, label: str) -> int:
    n = process_warc(warc_path, label)
    data_volume.commit()
    return n

@app.local_entrypoint()
def main_parallel():
    # TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    # TRAIN_PATH.write_text("")
    jobs = [
        # ("/root/data/quality_training/warc_wiki_chunk_aa.warc.warc.gz", "wiki"),
        ("/root/data/quality_training/warc_wiki_chunk_ab.warc.warc.gz", "wiki"),
        ("/root/data/quality_training/warc_wiki_chunk_ac.warc.warc.gz", "wiki"),
        ("/root/data/quality_training/warc_wiki_chunk_ad.warc.warc.gz", "wiki"),
        ("/root/data/quality_training/warc_wiki_chunk_ae.warc.warc.gz", "wiki")
    ]
    counts = []
    for warc_path, label in jobs:
        n = process_warc_modal.remote(warc_path, label)
        counts.append(n)
        print(warc_path, n)

    print("total lines:", sum(counts))
'''

def classify_quality(text: str) -> tuple[str, float]:
    # path = get_shared_assets_path() / "classifiers" / "quality_model.bin"
    path = "local-shared-data/classifiers/quality_model.bin"
    model = fasttext.load_model(str(path))
    text = text.replace("\n", " ")
    labels, probs = model.predict(text)
    label = labels[0].removeprefix("__label__")
    return label, float(probs[0])

# if __name__ == "__main__":
    # n = 0
    # with open('/Users/jenniferwang/Classes/cs336/assignment4-data/train.txt', 'a') as train_out:
    #     with open("local-shared-data/CC/example.warc.gz", "rb") as f:
    #         for record in ArchiveIterator(f, parse_http=True):
    #             if record.record_type == WarcRecordType.response:
    #                 html_content = record.reader.read()
    #                 text = extract_text_from_html_bytes(html_content)
    #                 record_id=record.headers["WARC-Record-ID"]
    #                 if n > 170:
    #                     break
    #                 if not text:
    #                     continue
    #                 if identify_language(text)[0] != "en":
    #                     continue
    #                 line = f"__label__cc {text.replace(chr(10), ' ')}\n"
    #                 train_out.write(line)
    #                 n += 1
                    

    # model = fasttext.train_supervised(
    #     input="/Users/jenniferwang/Classes/cs336/assignment4-data/train.txt",
    #     epoch=25,
    #     lr=1.0,
    #     wordNgrams=2,
    #     dim=64,
    # )
    # model.save_model("local-shared-data/classifiers/quality_model.bin")


