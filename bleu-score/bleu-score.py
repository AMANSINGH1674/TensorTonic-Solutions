import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    Returns: A float representing the BLEU score (0.0 to 1.0).
    """
    c_len = len(candidate)
    r_len = len(reference)
    
    # Edge case: If the candidate is empty, the score is 0.0
    if c_len == 0:
        return 0.0
        
    precisions = []
    
    # Iterate through all n-gram orders from 1 to max_n
    for n in range(1, max_n + 1):
        # Generate n-grams for candidate and reference
        # We use tuples because they are hashable and can be counted via Counter
        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(c_len - n + 1))
        ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(r_len - n + 1))
        
        # Calculate the total number of candidate n-grams of order `n`
        total_cand_ngrams = sum(cand_ngrams.values())
        
        # If there are no n-grams for this order, the precision is 0 (e.g., n > c_len)
        if total_cand_ngrams == 0:
            return 0.0
            
        # Calculate the clipped count: min(candidate_count, reference_count)
        clipped_count = 0
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            
        # If any precision is 0, the overall BLEU score becomes 0
        if clipped_count == 0:
            return 0.0
            
        precisions.append(clipped_count / total_cand_ngrams)
        
    # Combine precisions using a uniform-weight geometric mean
    # exp( (1 / max_n) * sum(log(p_n)) )
    log_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_sum / max_n)
    
    # Calculate the Brevity Penalty (BP)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r_len / c_len))
        
    # Final BLEU score
    return bp * geometric_mean
    