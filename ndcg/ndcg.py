import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k for a list of relevance scores.
    """
    # Edge case: empty list or invalid k
    if not relevance_scores or k <= 0:
        return 0.0

    # Ensure k doesn't exceed the number of available items
    k = min(k, len(relevance_scores))

    def compute_dcg(scores):
        dcg = 0.0
        for i in range(len(scores)):
            # Exponential gain places high value on highly relevant items
            gain = (2 ** scores[i]) - 1
            
            # Logarithmic discount penalizes items appearing lower in the ranking
            # Note: i is 0-indexed in Python, so position is i+1. 
            # The formula requires log2(position + 1), which becomes log2(i + 2).
            discount = math.log2(i + 2)
            
            dcg += gain / discount
        return dcg

    # 1. Compute the actual DCG for the top-k items
    actual_dcg = compute_dcg(relevance_scores[:k])

    # 2. Compute the Ideal DCG (IDCG) by sorting the entire list descending first
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = compute_dcg(ideal_scores)

    # Edge case: If all relevant scores are 0, IDCG is 0. Avoid division by zero.
    if idcg == 0.0:
        return 0.0

    # 3. Normalize to get NDCG
    return actual_dcg / idcg