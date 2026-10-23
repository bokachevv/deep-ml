import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    
    X = np.array(points)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        labels = np.array([np.argmin(np.linalg.norm(p-centroids, axis = 1)) for p in X])
        for i in range(len(centroids)):
            centroids[i] = np.mean(X[labels == i], axis = 0)
    final_centroids = np.round(centroids, 4)
    return [tuple(c) for c in final_centroids.tolist()]