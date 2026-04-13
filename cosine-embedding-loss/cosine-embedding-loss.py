import math

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # 1. Compute the dot product
    dot_product = sum(a * b for a, b in zip(x1, x2))
    
    # 2. Compute the L2 norms (magnitude) of both vectors
    norm1 = math.sqrt(sum(a * a for a in x1))
    norm2 = math.sqrt(sum(b * b for b in x2))
    
    # 3. Calculate cosine similarity
    cos_sim = dot_product / (norm1 * norm2)
    
    # 4. Apply the loss function based on the label
    if label == 1:
        return 1.0 - cos_sim
    elif label == -1:
        return max(0.0, cos_sim - margin)

# --- Test Cases ---
if __name__ == "__main__":
    # Example 1: Identical vectors (Similar)
    print(cosine_embedding_loss([1, 0, 0], [1, 0, 0], 1, 0.0))  # Output: 0.0
    
    # Example 2: Orthogonal vectors (Similar - model is penalized)
    print(cosine_embedding_loss([1, 0, 0], [0, 1, 0], 1, 0.0))  # Output: 1.0