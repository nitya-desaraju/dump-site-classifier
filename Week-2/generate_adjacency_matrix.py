import numpy as np

def generate_adjacency_matrix(dists: np.ndarray, C: float) -> np.ndarray:
    adjs = np.where(dists < C, 1 - dists, 0.0)
    return adjs
