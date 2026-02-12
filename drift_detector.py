import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FréchetDistanceDriftDetector:
    def __init__(self, threshold_multiplier=2.0, n_components=10):
        """
        DRIFTLENS Implementation: Fréchet Distance with PCA and Prototype Explanations.
        """
        self.threshold_multiplier = threshold_multiplier
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.reference_mean = None
        self.reference_cov = None
        self.threshold = None
        self.is_fitted = False
    
    def compute_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        # Implementation of Wasserstein-2 distance between two Gaussians
        diff = mu1 - mu2
        # Use sqrtm on product of covariances
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        distance = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return max(0, distance) # Ensure distance is non-negative
    
    def fit_reference(self, embeddings):
        """Reduces dimensions and fits reference Gaussian (Offline Phase)."""
        # DRIFTLENS Requirement: Dimension reduction for stability
        reduced_embeddings = self.pca.fit_transform(embeddings)
        
        self.reference_mean = np.mean(reduced_embeddings, axis=0)
        self.reference_cov = np.cov(reduced_embeddings.T)
        
        # Estimate Threshold using sub-windows (Offline Phase logic)
        n_samples = reduced_embeddings.shape[0]
        sub_window_size = max(n_samples // 10, 20)
        
        distances = []
        for i in range(0, n_samples - sub_window_size, sub_window_size):
            window = reduced_embeddings[i:i + sub_window_size]
            if len(window) > 1:
                mu_w = np.mean(window, axis=0)
                cov_w = np.cov(window.T)
                dist = self.compute_frechet_distance(self.reference_mean, self.reference_cov, mu_w, cov_w)
                distances.append(dist)
        
        self.threshold = np.mean(distances) + self.threshold_multiplier * np.std(distances)
        self.is_fitted = True

    def detect_global_drift(self, current_embeddings):
        if not self.is_fitted: raise ValueError("Fit the detector first.")
        
        # Transform window to the same PCA space
        reduced_w = self.pca.transform(current_embeddings)
        mu_w = np.mean(reduced_w, axis=0)
        cov_w = np.cov(reduced_w.T)
        
        drift_score = self.compute_frechet_distance(self.reference_mean, self.reference_cov, mu_w, cov_w)
        is_drift = drift_score > self.threshold
        return drift_score, is_drift, self.threshold

    def detect_classwise_drift(self, embeddings, predictions):
        """Identifies which class is most affected by drift."""
        reduced_embeddings = self.pca.transform(embeddings)
        unique_classes = np.unique(predictions)
        class_drift_scores = {}
        
        for cls in unique_classes:
            mask = (predictions == cls)
            cls_embs = reduced_embeddings[mask]
            if len(cls_embs) > 1:
                mu_cls = np.mean(cls_embs, axis=0)
                cov_cls = np.cov(cls_embs.T)
                class_drift_scores[int(cls)] = self.compute_frechet_distance(self.reference_mean, self.reference_cov, mu_cls, cov_cls)
            else:
                class_drift_scores[int(cls)] = 0.0
        
        most_affected = max(class_drift_scores, key=class_drift_scores.get) if class_drift_scores else None
        return class_drift_scores, most_affected

    def get_drift_explanation_prototypes(self, embeddings, most_affected_class, predictions, n_prototypes=3):
        """
        DRIFTLENS Prototype Explanation using K-Means clustering.
        Identifies representative samples for the drifted data.
        """
        if most_affected_class is None: return []
        
        # Filter embeddings for the affected class
        class_mask = (predictions == most_affected_class)
        class_embeddings = embeddings[class_mask]
        original_indices = np.where(class_mask)[0]
        
        if len(class_embeddings) < n_prototypes: return original_indices.tolist()
        
        # Cluster the drifted embeddings to find distinct drift "concepts"
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42).fit(class_embeddings)
        centroids = kmeans.cluster_centers_
        
        prototype_indices = []
        for center in centroids:
            # Find the sample closest to the cluster center (the prototype)
            dist = np.linalg.norm(class_embeddings - center, axis=1)
            closest_idx = np.argmin(dist)
            prototype_indices.append(int(original_indices[closest_idx]))
            
        return prototype_indices