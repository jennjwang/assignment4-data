"""
 Write a function that will take a Unicode string and identify the main language that is
present in this string. Your function should return a pair, containing an identifier of the
language and a score between 0 and 1 representing its confidence in that prediction.
Deliverable: A function that performs language identification, giving its top language
prediction and a score. Implement the adapter [run_identify_language] and make sure it
passes both tests in uv run pytest -k test_identify_language. Note that these tests assume
a particular string identifier for English (“en”) and Chinese (“zh”), so your test adapter
should perform any applicable re-mapping, if necessary.
"""

import fasttext
from cs336_data.common import get_shared_assets_path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes

def identify_language(text: str) -> tuple[str, float]:
    p = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    # print(p)

    model = fasttext.load_model(str(p))

    text = text.replace("\n", "")
    labels, probabilities = model.predict(text)
    
    label = labels[0]
    # print(repr(label), type(label))
    code = label.removeprefix("__label__")

    return code, probabilities[0]

"""
Run your language identification system on text extracted from the WARC files (via your
previously-implemented text extraction function). Manually identify the language in 20
random examples and compare your labels with the classifier predictions. Report any classifier
errors. What fraction of documents are English? Based on your observations, what would be a
suitable classifier confidence threshold to use in filtering?
"""

if __name__ == "__main__":
    # print(extract_text_from_html_bytes("café <p>hello</p>".encode("latin-1")))
    # print(resiliparse.parse.encoding.detect_encoding(b"\x00\x01\x02"))
    i = 0

    import os
    os.makedirs("language_identification_review", exist_ok=True)

    with open("local-shared-data/CC/example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, parse_http=True):
            if record.record_type == WarcRecordType.response:
                html_content = record.reader.read()
                text = extract_text_from_html_bytes(html_content)
                lang, score = identify_language(text)
                path = f"language_identification_review/record_{i:02d}_{lang}_{score:.4f}.txt"
                with open(path, "w", encoding="utf-8") as out:
                    out.write(text)
                i += 1
            if i == 20:
                break
                
    """
    0 ('zh', 0.8964954614639282)
    1 ('zh', 0.9806137084960938)
    2 ('zh', 0.9039925932884216)
    3 ('zh', 0.9965595602989197)
    4 ('zh', 0.9889840483665466)
    5 ('zh', 0.9577594995498657)
    6 ('zh', 0.953950047492981)
    7 ('en', 0.6947742700576782)
    8 ('ru', 0.9933364987373352)
    9 ('ru', 0.9773629307746887)
    10 ('de', 0.9160632491111755)
    11 ('zh', 0.9926297068595886)
    12 ('zh', 0.8028445243835449)
    13 ('el', 0.9982876777648926)
    14 ('en', 0.8620272278785706)
    15 ('zh', 0.7337842583656311)
    16 ('zh', 0.9475809335708618)
    17 ('zh', 0.8532630205154419)
    18 ('en', 0.11044929921627045)
    19 ('en', 0.9532276391983032)
    """
# print(identify_language("hello world"))

