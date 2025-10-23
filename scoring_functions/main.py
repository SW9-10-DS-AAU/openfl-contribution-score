# main.py
import numpy as np

# Import scoring functions
from cosine_score import cosine_score
from dot_score import dot_score
from fedavg_dot_score import fedavg_dot_score

# Example client weights (or updates)
# Each client update is a numpy array representing the difference from global model
client_updates = [
    np.array([0.1, 0.2, 0.3]),   # Client 1
    np.array([0.05, -0.1, 0.2]), # Client 2
    np.array([-0.1, 0.05, 0.0])  # Client 3
]

# Compute global update using FedAvg (average)
U = sum(client_updates) / len(client_updates)

# Compute scores
cosine_scores = [cosine_score(u, U) for u in client_updates]
dot_scores = [dot_score(u, U) for u in client_updates]
fedavg_dot_scores = [fedavg_dot_score(u, U, len(client_updates)) for u in client_updates]

# Print results
print("Global update:", U)

print("\nCosine similarity scores:")
for i, s in enumerate(cosine_scores):
    print(f"Client {i+1}: {s:.4f}")
print("Sum of Cosine-simularity scores:", sum(cosine_scores))

print("\nRaw dot product scores:")
for i, s in enumerate(dot_scores):
    print(f"Client {i+1}: {s:.4f}")
print("Sum of dot-product scores:", sum(dot_scores))

print("\nFedAvg-normalized dot product scores (sum=1):")
for i, s in enumerate(fedavg_dot_scores):
    print(f"Client {i+1}: {s:.4f}")
print("Sum of FedAvg-normalized scores:", sum(fedavg_dot_scores))
