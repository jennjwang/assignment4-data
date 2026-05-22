"""
Write a function to mask out emails. Your function will take a string as input, and replace all
instances of email addresses with the string "|||EMAIL_ADDRESS|||". To detect email
addresses, you can look up regular expressions that do this reliably.
Deliverable: A function that replaces all email addresses in a given string with the string
"|||EMAIL_ADDRESS|||", returning a pair containing both the new string and the number of
instances that were masked. Implement the adapter [run_mask_emails] and make sure it
passes all tests in uv run pytest -k test_mask_emails.
"""

import re
import fasttext
from cs336_data.common import get_shared_assets_path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.extract_text import extract_text_from_html_bytes


def mask_emails(text):
    masked_text, num_masked = re.subn(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "|||EMAIL_ADDRESS|||",
        text,
    )
    return (masked_text, num_masked)

"""
A function that replaces phone numbers in a given string with the string "|||
PHONE_NUMBER|||", returning a pair containing both the new string and the number of
instances that were masked. Implement the adapter [run_mask_phone_numbers] and make sure
it passes uv run pytest -k test_mask_phones
"""

def mask_phones(text):
    masked_text, num_masked = re.subn(
        r"(?:\(\d{3}\)[-\s]?\d{3}[-\s]?\d{4}|\d{3}-\d{3}-\d{4}|\b\d{10}\b)",
        "|||PHONE_NUMBER|||",
        text,
    )
    return (masked_text, num_masked)

"""
A function that replaces IPv4 addresses in a given string with the string "|||
IP_ADDRESS|||", returning a pair containing both the new string and the number of instances
that were masked. Implement the adapter [run_mask_ips] and make sure it passes uv run
pytest -k test_mask_ips
"""

def mask_ips(text):
    masked_text, num_masked = re.subn(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "|||IP_ADDRESS|||",
        text,
    )
    return (masked_text, num_masked)

if __name__ == "__main__":
    # print(extract_text_from_html_bytes("café <p>hello</p>".encode("latin-1")))
    # print(resiliparse.parse.encoding.detect_encoding(b"\x00\x01\x02"))
    i = 0

    with open("local-shared-data/CC/example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, parse_http=True):
            if record.record_type == WarcRecordType.response:
                record_id=record.headers["WARC-Record-ID"]
                html_content = record.reader.read()
                text = extract_text_from_html_bytes(html_content)
                total_masked = 0
                masked_text, num_masked = mask_ips(text)
                total_masked += num_masked
                masked_text, num_masked = mask_phones(masked_text)
                total_masked += num_masked
                masked_text, num_masked = mask_emails(masked_text)
                total_masked += num_masked
                if total_masked > 0:
                    i += 1
                    with open(f"example_{record_id}.txt", "w") as f:
                        f.write(text)
            if i == 20:
                break
            