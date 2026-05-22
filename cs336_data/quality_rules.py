"""
For this assignment, you will implement a subset of the filters described in the
Gopher paper [2]. Specifically, you should remove documents that:
• Contain less than 50 or more than 100,000 words.
• Have a mean word length outside the range of 3 to 10 characters.
• Have more than 30% of lines ending with an ellipsis (“...”).
• Contain less than 80% of words with at least one alphabetic character.
"""

from nltk.tokenize import word_tokenize
import fasttext
from cs336_data.common import get_shared_assets_path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes
import random


def gopher_filters(text):

    words = word_tokenize(text)
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False
    mean_length = sum(len(w) for w in words) / len(words)
    if mean_length < 3 or mean_length > 10:
        return False
    lines = text.splitlines()
    ellipsis_pcnt = sum([line.endswith("...") for line in lines]) / len(lines)
    if ellipsis_pcnt > 0.3:
        return False
    alpha_pcnt = sum(any(c.isalpha() for c in w) for w in words) / len(words)
    if alpha_pcnt < 0.8:
        return False

    return True

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
        label = gopher_filters(text)
        with open(f"results/gopher/{label}_example_{record_id}.txt", "w") as f:
            f.write(text)
