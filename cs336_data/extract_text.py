'''
A function that takes a byte string containing HTML and returns a string
containing the extracted text. Implement the adapter [run_extract_text_from_html_bytes]
and make sure it passes uv run pytest -k test_extract_text_from_html_bytes
'''

from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse
import resiliparse.parse.encoding
import resiliparse.extract.html2text

def extract_text(html_content):
    return resiliparse.extract.html2text.extract_plain_text(html_content)

def detect_encoding(html_content):
    return resiliparse.parse.encoding.detect_encoding(html_content)

def extract_text_from_html_bytes(html_content):
    try:
        unicode_html_content = html_content.decode("utf-8")
        text = extract_text(unicode_html_content)
        return text
    except UnicodeDecodeError:
        encoding = detect_encoding(html_content)
        unicode_html_content = html_content.decode(encoding, errors="replace")
        text = extract_text(unicode_html_content)
        return text

if __name__ == "__main__":
    print(extract_text_from_html_bytes("café <p>hello</p>".encode("latin-1")))
    print(resiliparse.parse.encoding.detect_encoding(b"\x00\x01\x02"))
    i = 0

    with open("local-shared-data/CC/example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, parse_http=True):
            if record.record_type == WarcRecordType.response:
                html_content = record.reader.read()
                text = extract_text_from_html_bytes(html_content)
                record_id=record.headers["WARC-Record-ID"]
                if i > 5:
                    break
                with open(f"example_{record_id}.txt", "w") as f:
                    f.write(text)
                    i += 1