"""
Write a function that takes a list of paths to input files and performs exact line deduplication on
them. It should first count the frequency of each line in the corpus, using a hash to reduce
memory, and then rewrite each file by only keeping its unique lines.

Deliverable: A function that performs exact line deduplication. Your function should take two
arguments: (a) a list of paths to its input files, and (b) an output directory. It should rewrite each
input file to the output directory with the same name, but deduplicate the content by removing
lines that occur more than once in the set of input files. For example, if the input paths are
a/1.txt and a/2.txt, and the output directory is b/, your function should write the files b/1.txt
and b/2.txt. Implement the adapter [run_exact_line_deduplication] and make sure it passes uv
run pytest -k test_exact_line_deduplication
"""

from pathlib import Path
import random
import itertools
import unicodedata
import string
import math

def exact_dedup(inputs, output):
    hash_freq = {}
    for p in inputs:
        with open(p) as file:
            lines = file.readlines()
            for line in lines:
                hashed = hash(line)
                hash_freq[hashed] = hash_freq.get(hashed, 0) + 1
        
    for p in inputs:
        with open(p) as file:
            lines = file.readlines()
            new_lines = []
            for line in lines:
                hashed = hash(line)
                if hash_freq[hashed] == 1:
                    new_lines.append(line)
        
        output_path = Path(output) / Path(p).name
        with open(output_path, 'w') as output_f:
            output_f.writelines(new_lines)
        

"""
Write a function that takes a list of paths to input files and performs fuzzy document
deduplication with minhash and LSH. In particular, your function should compute minhash
signatures for each document in the provided list of paths, use LSH with the provided number of
bands to identify candidate duplicates, and then compute the true ngram Jaccard similarity
between candidate duplicates and remove those that exceed a given threshold. To improve recall
(following [7]), normalize the text before computing minhash signatures and/or comparing
Jaccard similarity by lowercasing, removing punctuation, normalizing whitespaces, and removing
accents, and applying NFD unicode normalization.

Deliverable: A function that performs fuzzy document deduplication. Your function should take
at least the following arguments: (a) a list of paths to its input files, (b) the number of hashes to
use for computing minhash signatures, (c) the number of bands to use for LSH, (d) the n-gram
length (in words) for computing minhash signatures, and (e) an output directory. You may 
assume that the number of hashes to use for computing minhash signatures is evenly divisible by
the number of bands to use for LSH.

Your function should rewrite each input file to the output directory with the same name, but only
writing documents that are either (a) not candidate duplicates and/or (b) are randomly selected
to be retained from the clustered buckets. For example, if the input paths are a/1.txt and
a/2.txt, and the output directory is b/, your function should write the files b/1.txt and b/2.txt.
Implement the adapter [run_minhash_deduplication] and make sure it passes uv run pytest -k
test_minhash_deduplication.
"""

def minhash(n_grams, num_hashes):
    hashes = []
    for i in range(num_hashes):
        min_hash = math.inf
        for g in n_grams:
            hashed = hash((g, i))
            if hashed < min_hash:
                min_hash = hashed
        hashes.append(min_hash)
    return hashes

def lsh(num_bands, num_hashes, doc_to_hash):
    band_bucket = {}
    band_len = num_hashes // num_bands
    for doc, hashed in doc_to_hash.items():
        bands = [hashed[i*band_len : (i+1)*band_len] for i in range(num_bands)]
        for band in bands:
            band_key = tuple(band)
            band_bucket.setdefault(band_key, []).append(doc)
    
    candidates = set()
    for band, docs in band_bucket.items():
        if len(docs) > 1:
            pairs = itertools.combinations(docs, 2)
            candidates.update(pairs)
    
    return candidates

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

def minihash_dedup(input_files, num_hashes, num_bands, n_gram_len, output_directory, threshold):

    doc_to_hash = {}
    doc_to_ngrams = {}
    for p in input_files:
        with open(p) as file:
            text = file.read()
            words = normalize(text)
            n_grams = [tuple(words[i: (i + n_gram_len)]) for i in range(len(words) - n_gram_len + 1)]
            minhashes = minhash(n_grams, num_hashes)
            doc_to_hash[p] = minhashes
            doc_to_ngrams[p] = set(n_grams)
    
    to_remove = set()
    candidate_pairs = lsh(num_bands, num_hashes, doc_to_hash)
    for pair in candidate_pairs:
        a, b = pair
        gram_a, gram_b = doc_to_ngrams[a], doc_to_ngrams[b]
        jaccard = len(gram_a & gram_b) / len(gram_a | gram_b)
        if jaccard > threshold:
            to_remove.add(random.choice(pair))
    
    for p in input_files:
        if p not in to_remove:
            with open(p) as file:
                text = file.read()
                output_path = Path(output_directory) / Path(p).name
                with open(output_path, 'w') as output_f:
                    output_f.write(text)
            

                
            

    