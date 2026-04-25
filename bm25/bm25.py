import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([])
        
    # Precompute document lengths and average document length
    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avgdl = np.mean(doc_lengths)
    
    # Precompute term frequencies (TF) and document frequencies (DF)
    # We use a list of Counters for fast per-document TF lookup
    doc_tfs = [Counter(doc) for doc in docs]
    
    # Calculate document frequency (DF) for each unique term in the query
    # We only care about query terms to save computation
    unique_query_tokens = list(dict.fromkeys(query_tokens))
    df = Counter()
    for doc in docs:
        unique_terms_in_doc = set(doc)
        for term in unique_query_tokens:
            if term in unique_terms_in_doc:
                df[term] += 1
                
    # Initialize the final score array
    scores = np.zeros(N, dtype=float)
    
    # Compute the BM25 score for each query term
    for term in unique_query_tokens:
        # Calculate IDF for the term
        # If term doesn't exist in corpus, df[term] is 0
        df_t = df[term]
        
        # Calculate standard Okapi BM25 IDF
        idf = math.log(((N - df_t + 0.5) / (df_t + 0.5)) + 1.0)
        
        # If IDF is negative (term appears in > half the documents), clamp to 0 
        # (Optional but common practice in Lucene/Elasticsearch to prevent negative scores)
        if idf < 0:
             idf = 0.0
             
        # Skip calculating TF and scoring if the term isn't in the corpus
        if idf == 0.0:
            continue
            
        # Extract the TF of the current term for all documents
        # tf_array shape: (N,)
        tf_array = np.array([doc_tf[term] for doc_tf in doc_tfs], dtype=float)
        
        # Calculate the TF normalization factor (numerator and denominator)
        # We only calculate this for documents where TF > 0 to avoid unnecessary math
        mask = tf_array > 0
        if not np.any(mask):
             continue
             
        # Calculate the denominator of the BM25 TF formula
        # tf + k1 * (1 - b + b * (doc_length / avgdl))
        denominator = tf_array[mask] + k1 * (1 - b + b * (doc_lengths[mask] / avgdl))
        
        # Calculate the numerator
        # tf * (k1 + 1)
        numerator = tf_array[mask] * (k1 + 1.0)
        
        # Add the score for this term to the total scores
        scores[mask] += idf * (numerator / denominator)
        
    return scores