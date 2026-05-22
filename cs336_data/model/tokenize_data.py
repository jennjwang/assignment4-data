import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
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

MINHASH_DEDUP_DIR = Path("/root/data/deduped/minhash-filtered-wet-text")

# input_path = "/root/data/filtered-wet-text"
# input_path = "/root/data/deduped/exact-filtered-wet-text"
input_path = str(MINHASH_DEDUP_DIR)
output_path = "/root/data/tokenized/train.bin"

# MODAL_OUT_DIR = Path("/root/data/filtered-wet-text")
# EXACT_DEDUP_DIR = Path("/root/data/deduped/exact-filtered-wet-text")
# MINHASH_DEDUP_DIR = Path("/root/data/deduped/minhash-filtered-wet-text")

# Tokenized and encoded /root/data/filtered-wet-text into 633,111,598 tokens
# Tokenized and encoded /root/data/deduped/exact-filtered-wet-text into 476,186,487 tokens
# Tokenized and encoded /root/data/deduped/minhash-filtered-wet-text into 95,421,432 tokens

# Tokenized and encoded /root/data/deduped/exact-filtered-wet-text into 163,388,679 tokens
# Tokenized and encoded /root/data/filtered-wet-text into 747668617 tokens
# Tokenized and encoded /root/data/deduped/minhash-filtered-wet-text into 705837793 tokens
# Tokenized and encoded /root/data/deduped/minhash-filtered-wet-text into 701,198,301 tokens
# Tokenized and encoded /root/data/deduped/exact-filtered-wet-text into 587326847 tokens
# Tokenized and encoded /root/data/deduped/minhash-filtered-wet-text into 583,924,851 tokens
@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 12,
)
def run_tokenize():
    results = []
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    for txt_path in Path(input_path).glob("*.txt"):
        text = txt_path.read_text()
        docs = text.split("\n<|endoftext|>\n")
        chunksize = 100

        batch_ids = tokenizer(docs, add_special_tokens=False)["input_ids"]
        for doc_id in batch_ids:
            doc_id.append(tokenizer.eos_token_id)
        results.extend(batch_ids)

        # for result in tqdm(
        #     tokenize_line_and_add_eos.map(docs, chunksize=chunksize),
        #     total=len(docs),
        #     desc="Tokenizing lines"
        # ):
        #     results.append(result)
    
    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
    # ids_array = np.array(all_ids, dtype=np.uint16)
    # Path("/root/data/tokenized").mkdir(parents=True, exist_ok=True)
    # ids_array.tofile(output_path)
    # data_volume.commit()

@app.local_entrypoint()
def modal_run_tokenize():
    run_tokenize.remote()
    
    # with open(input_path) as f:
    #     lines = f.readlines()
    #     pool = multiprocessing.Pool(multiprocessing.cpu_count())
    #     chunksize = 100
    #     results = []
        
        # for result in tqdm(
        #     pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
        #     total=len(lines),
        #     desc="Tokenizing lines"):
        #     results.append(result)
        #     pool.close()
        #     pool.join()
    
        # # Flatten the list of ids and convert to numpy array
        # all_ids = [token_id for sublist in results for token_id in sublist]
        # print(f"Tokenized and encoded {input_path} into {len(all_ids)} tokens")
        # ids_array = np.array(all_ids, dtype=np.uint16)
        # ids_array.tofile(output_path)