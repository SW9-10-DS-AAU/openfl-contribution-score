import numpy as np


def fedavg_dot_score(u, U, num_clients, eps=1e-12):
    """
    FedAvg-normalized dot product score so that sum = 1.

    Args:
        u: numpy array, client update
        U: numpy array, global update (FedAvg)
        num_clients: int, number of clients in aggregation
        eps: float, small tolerance for zero norm

    Returns:
        float, contribution score
    """
    norm_U_sq = np.dot(U, U)

    if norm_U_sq < eps:
        # Global update is effectively zero, no contribution
        return 0.0
    else:
        return np.dot(u, U) / (num_clients * norm_U_sq)
