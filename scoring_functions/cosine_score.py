import numpy as np

def cosine_score(u, U):
    return np.dot(u, U) / (np.linalg.norm(u) * np.linalg.norm(U) + 1e-12)
