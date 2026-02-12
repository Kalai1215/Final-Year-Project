import numpy as np
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

class DriftEngine:
    def __init__(self, n_components=10, threshold_multiplier=2.0):
        """
        Core DRIFTLENS Engine
        n_components: d' for PCA reduction (improves stability)
        threshold_multiplier: Sensitivity for drift detection
        """
        self.n_components = n_components
        self.threshold_multiplier = threshold_multiplier
        self.pca = PCA(n_components=n_components)
        self.ref_mu = None
        self.ref_sigma = None
        self.threshold = None
        self.is_fitted = False

    def _calculate_fdd(self, mu1, sigma1, mu2, sigma2):
        """Calculates the Fréchet (Wasserstein-2) Distance"""
        diff = mu1 - mu2
        # Product of covariances for the trace term
        cov_prod = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real
        
        # FDD Formula: ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        distance = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * cov_prod)
        return max(0, distance)

    def fit_baseline(self, train_embeddings):
        """
        Offline Phase: Processes historical data to create a reference distribution.
        """
        # Step 1: Reduce dimensions
        reduced_embs = self.pca.fit_transform(train_embeddings)
        
        # Step 2: Compute Reference Stats
        self.ref_mu = np.mean(reduced_embs, axis=0)
        self.ref_sigma = np.cov(reduced_embs, rowvar=False)
        
        # Step 3: Estimate Threshold via sub-windowing
        n = len(reduced_embs)
        win_size = max(n // 10, 20)
        distances = []
        for i in range(0, n - win_size, win_size):
            win = reduced_embs[i : i + win_size]
            mu_w = np.mean(win, axis=0)
            sigma_w = np.cov(win, rowvar=False)
            distances.append(self._calculate_fdd(self.ref_mu, self.ref_sigma, mu_w, sigma_w))
        
        self.threshold = np.mean(distances) + (self.threshold_multiplier * np.std(distances))
        self.is_fitted = True
        print(f"Baseline fitted. Threshold set at: {self.threshold:.4f}")

    def detect_drift(self, window_embeddings):
        """
        Online Phase: Compares current window to baseline.
        """
        if not self.is_fitted:
            raise ValueError("Engine must be fitted with baseline data first.")
            
        # Reduce window using the same PCA
        reduced_w = self.pca.transform(window_embeddings)
        mu_w = np.mean(reduced_w, axis=0)
        sigma_w = np.cov(reduced_w, rowvar=False)
        
        distance = self._calculate_fdd(self.ref_mu, self.ref_sigma, mu_w, sigma_w)
        is_drift = distance > self.threshold
        
        return distance, is_drift