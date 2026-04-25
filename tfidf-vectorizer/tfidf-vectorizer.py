import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    N = len(documents)
    
    # Handle empty corpus
    if N == 0:
        return np.empty((0, 0)), []
        
    tokenized_docs = []
    df_counter = Counter()
    
    # 1. Tokenize and compute Document Frequency (DF)
    for doc in documents:
        # Tokenize by splitting on whitespace and converting to lowercase
        tokens = doc.lower().split()
        tokenized_docs.append(tokens)
        
        # Keep track of unique words in this document for DF
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df_counter[token] += 1
            
    # 2. Build Vocabulary
    vocabulary = sorted(df_counter.keys())
    V = len(vocabulary)
    
    # Handle corpus with entirely empty documents
    if V == 0:
        return np.zeros((N, 0)), []
        
    # Map words to their column indices for fast matrix population
    word2idx = {word: idx for idx, word in enumerate(vocabulary)}
    
    # 3. Precompute IDF array
    # idf(t) = log(N / df(t))
    idf_array = np.array([math.log(N / df_counter[word]) for word in vocabulary])
    
    # 4. Compute Term Frequency (TF) matrix
    tf_matrix = np.zeros((N, V), dtype=float)
    
    for i, tokens in enumerate(tokenized_docs):
        total_terms = len(tokens)
        if total_terms == 0:
            continue
            
        # Count frequencies in the current document
        counts = Counter(tokens)
        for word, count in counts.items():
            # tf(t, d) = count(t, d) / total_terms
            tf_matrix[i, word2idx[word]] = count / total_terms
            
    # 5. Compute final TF-IDF matrix
    # Vectorized multiplication using NumPy broadcasting: (N, V) * (V,)
    tfidf_matrix = tf_matrix * idf_array
    
    return tfidf_matrix, vocabulary