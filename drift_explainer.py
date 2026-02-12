import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class DriftExplainer:
    def __init__(self, max_prototypes=5):
        self.max_prototypes = max_prototypes

    def explain(self, drifted_embeddings):
        """
        Identifies representative prototypes from the drifted window.
        Uses K-Means and Silhouette Score to find distinct drift patterns.
        """
        n_samples = len(drifted_embeddings)
        
        # Find optimal K (number of drift concepts)
        best_k = 1
        if n_samples > 10:
            best_score = -1
            for k in range(2, self.max_prototypes + 1):
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(drifted_embeddings)
                score = silhouette_score(drifted_embeddings, km.labels_)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Perform final clustering
        final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(drifted_embeddings)
        centroids = final_km.cluster_centers_
        
        # Select one prototype per cluster (sample closest to centroid)
        prototype_indices = []
        for i in range(best_k):
            center = centroids[i]
            # Euclidean distance to find the representative sample
            dists = np.linalg.norm(drifted_embeddings - center, axis=1)
            prototype_indices.append(np.argmin(dists))
            
        return prototype_indices # Indices of samples to show for explanation