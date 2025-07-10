import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    if np.any(arr1) and np.any(arr2):
        return 1 - cosine(arr1, arr2)
    return 0.0


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0


def epsilon_greedy(candidates: List[Tuple[int, float]], epsilon: float) -> int:
    """
    candidates: list of (candidate_id, score)
    returns chosen candidate_id
    """
    import random
    if random.random() < epsilon:
        return random.choice(candidates)[0]
    return max(candidates, key=lambda x: x[1])[0]
