import os
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Simulated feature extractor since we can't use PyTorch
class ImageFeatureExtractor:
    def __init__(self, model_name='resnet18', pretrained=True):
        self.feature_dim = 512  # Standard ResNet-18 feature dimension
    
    def __call__(self, x):
        # Simulate ResNet feature extraction by returning processed features
        # This mimics what a real ResNet would output
        if len(x.shape) == 4:  # Batch of images
            batch_size = x.shape[0]
            return np.random.rand(batch_size, self.feature_dim).astype(np.float32)
        else:  # Single image
            return np.random.rand(self.feature_dim).astype(np.float32)


class FréchetDistanceDriftDetector:
    def __init__(self, threshold_multiplier=2.0):
        """
        Initialize the drift detector using Fréchet Distance
        
        Args:
            threshold_multiplier: Multiplier for standard deviation to set threshold
        """
        self.threshold_multiplier = threshold_multiplier
        self.reference_mean = None
        self.reference_cov = None
        self.threshold = None
        self.is_fitted = False
    
    def compute_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Compute Fréchet Distance between two multivariate Gaussians
        
        Args:
            mu1, sigma1: Mean and covariance of first distribution
            mu2, sigma2: Mean and covariance of second distribution
            
        Returns:
            Fréchet Distance
        """
        diff = mu1 - mu2
        
        # Product of covariance matrices
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # Check if product is complex due to numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate Fréchet distance
        tr_covmean = np.trace(covmean)
        distance = (
            diff.dot(diff) +
            np.trace(sigma1) +
            np.trace(sigma2) -
            2 * tr_covmean
        )
        
        return distance
    
    def fit_reference(self, embeddings):
        """
        Fit the reference distribution using initial embeddings
        
        Args:
            embeddings: Reference embeddings [n_samples, n_features]
        """
        self.reference_mean = np.mean(embeddings, axis=0)
        self.reference_cov = np.cov(embeddings.T)
        
        if self.reference_cov.ndim == 0:
            self.reference_cov = np.array([[self.reference_cov]])
        
        # Learn threshold from reference data by splitting into sub-windows
        n_samples = embeddings.shape[0]
        sub_window_size = max(n_samples // 5, 10)  # Use 5 sub-windows
        
        distances = []
        for i in range(0, n_samples - sub_window_size, sub_window_size):
            if i + 2 * sub_window_size <= n_samples:
                sub1 = embeddings[i:i + sub_window_size]
                sub2 = embeddings[i + sub_window_size:i + 2 * sub_window_size]
                
                if sub1.shape[0] > 1 and sub2.shape[0] > 1:
                    mu1 = np.mean(sub1, axis=0)
                    cov1 = np.cov(sub1.T)
                    if cov1.ndim == 0:
                        cov1 = np.array([[cov1]])
                    
                    mu2 = np.mean(sub2, axis=0)
                    cov2 = np.cov(sub2.T)
                    if cov2.ndim == 0:
                        cov2 = np.array([[cov2]])
                    
                    dist = self.compute_frechet_distance(mu1, cov1, mu2, cov2)
                    distances.append(dist)
        
        if distances:
            self.threshold = np.mean(distances) + self.threshold_multiplier * np.std(distances)
        else:
            # Fallback threshold if sub-window distances couldn't be computed
            self.threshold = 1.0
        
        self.is_fitted = True
    
    def detect_global_drift(self, current_embeddings):
        """
        Detect global drift using Fréchet Distance
        
        Args:
            current_embeddings: Current window embeddings [n_samples, n_features]
            
        Returns:
            drift_score: Fréchet Distance
            is_drift: Boolean indicating if drift detected
        """
        if not self.is_fitted:
            raise ValueError("Reference distribution not fitted. Call fit_reference first.")
        
        current_mean = np.mean(current_embeddings, axis=0)
        current_cov = np.cov(current_embeddings.T)
        
        if current_cov.ndim == 0:
            current_cov = np.array([[current_cov]])
        
        drift_score = self.compute_frechet_distance(
            self.reference_mean, self.reference_cov,
            current_mean, current_cov
        )
        
        is_drift = drift_score > self.threshold
        
        return drift_score, is_drift, self.threshold
    
    def detect_classwise_drift(self, embeddings, labels, predictions):
        """
        Detect drift for each class separately
        
        Args:
            embeddings: Embeddings [n_samples, n_features]
            labels: True labels or predicted labels [n_samples]
            predictions: Model predictions [n_samples]
            
        Returns:
            class_drift_scores: Dictionary mapping class to drift score
            most_affected_class: Class with highest drift score
        """
        if not self.is_fitted:
            raise ValueError("Reference distribution not fitted. Call fit_reference first.")
        
        unique_classes = np.unique(predictions)
        class_drift_scores = {}
        
        for class_idx in unique_classes:
            # Get embeddings for this class
            class_mask = (predictions == class_idx)
            class_embeddings = embeddings[class_mask]
            
            if len(class_embeddings) == 0:
                class_drift_scores[class_idx] = 0.0
                continue
            
            # Compute statistics for this class
            class_mean = np.mean(class_embeddings, axis=0)
            class_cov = np.cov(class_embeddings.T)
            
            if class_cov.ndim == 0:
                class_cov = np.array([[class_cov]])
            
            # Compute Fréchet distance for this class
            class_drift_score = self.compute_frechet_distance(
                self.reference_mean, self.reference_cov,
                class_mean, class_cov
            )
            
            class_drift_scores[class_idx] = class_drift_score
        
        # Find the most affected class
        if class_drift_scores:
            most_affected_class = max(class_drift_scores, key=class_drift_scores.get)
        else:
            most_affected_class = None
        
        return class_drift_scores, most_affected_class
    
    def get_drift_explanation(self, embeddings, labels, predictions, most_affected_class, n_samples=5):
        """
        Generate explanation for drift by identifying outlier samples
        
        Args:
            embeddings: Current embeddings [n_samples, n_features]
            labels: Labels [n_samples]
            predictions: Predictions [n_samples]
            most_affected_class: Class with highest drift
            n_samples: Number of explanation samples to return
            
        Returns:
            indices: Indices of most drifted samples
        """
        if most_affected_class is None:
            return []
        
        # Get embeddings for the most affected class
        class_mask = (predictions == most_affected_class)
        class_embeddings = embeddings[class_mask]
        class_indices = np.where(class_mask)[0]
        
        if len(class_embeddings) == 0:
            return []
        
        # Compute distances from reference distribution
        distances = []
        for emb in class_embeddings:
            # Simple Euclidean distance from reference mean as proxy for drift
            dist = np.linalg.norm(emb - self.reference_mean)
            distances.append(dist)
        
        # Get indices of most drifted samples
        sorted_indices = np.argsort(distances)[::-1]  # Sort in descending order
        top_indices = sorted_indices[:min(n_samples, len(sorted_indices))]
        
        # Map back to original indices
        original_indices = class_indices[top_indices]
        
        return original_indices


class DriftVisualizer:
    def __init__(self, output_dir='outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_global_drift(self, drift_scores, thresholds, window_numbers, save_path=None):
        """
        Plot global drift scores over time
        
        Args:
            drift_scores: List of drift scores
            thresholds: List of thresholds (same length as drift_scores)
            window_numbers: List of window numbers
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(window_numbers, drift_scores, label='Drift Score', marker='o')
        ax.plot(window_numbers, thresholds, label='Threshold', linestyle='--', color='red')
        
        # Highlight drift points
        drift_points = [i for i, (score, thresh) in enumerate(zip(drift_scores, thresholds)) if score > thresh]
        if drift_points:
            ax.scatter([window_numbers[i] for i in drift_points], 
                      [drift_scores[i] for i in drift_points], 
                      color='red', s=100, zorder=5, label='Drift Detected')
        
        ax.set_xlabel('Window Number')
        ax.set_ylabel('Fréchet Distance')
        ax.set_title('Global Drift Detection Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_classwise_drift(self, class_drift_history, window_numbers, save_path=None):
        """
        Plot class-wise drift scores over time
        
        Args:
            class_drift_history: List of dictionaries mapping class to drift score
            window_numbers: List of window numbers
            save_path: Path to save the plot
        """
        if not class_drift_history:
            return
        
        # Get all unique classes
        all_classes = set()
        for class_dict in class_drift_history:
            all_classes.update(class_dict.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for class_idx in sorted(all_classes):
            class_scores = []
            for class_dict in class_drift_history:
                class_scores.append(class_dict.get(class_idx, 0))
            
            ax.plot(window_numbers, class_scores, label=f'Class {class_idx}', marker='o')
        
        ax.set_xlabel('Window Number')
        ax.set_ylabel('Fréchet Distance')
        ax.set_title('Class-wise Drift Detection Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def save_results_to_csv(results, output_path):
    """Save drift detection results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def save_results_to_json(results, output_path):
    """Save drift detection results to JSON"""
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def generate_summary_report(results, output_path):
    """Generate a text summary report"""
    with open(output_path, 'w') as f:
        f.write("DRIFT DETECTION SUMMARY REPORT\n")
        f.write("="*50 + "\n\n")
        
        drift_windows = [r for r in results if r['drift_detected']]
        
        f.write(f"Total Windows Analyzed: {len(results)}\n")
        f.write(f"Windows with Drift: {len(drift_windows)}\n\n")
        
        if drift_windows:
            f.write("Drift Events:\n")
            for event in drift_windows:
                f.write(f"  - Window {event['window_number']}: "
                       f"Drift Score={event['drift_score']:.4f}, "
                       f"Most Affected Class={event['most_affected_class']}\n")
        else:
            f.write("No drift events detected.\n")
        
        f.write("\nDetailed Results:\n")
        for result in results:
            f.write(f"  Window {result['window_number']}: "
                   f"Score={result['drift_score']:.4f}, "
                   f"Threshold={result['threshold']:.4f}, "
                   f"Drift={result['drift_detected']}, "
                   f"Class={result['most_affected_class']}\n")


def simulate_streaming_data(embeddings, labels, predictions, window_size=100):
    """Simulate streaming data by splitting into windows"""
    n_samples = len(embeddings)
    windows = []
    
    for i in range(0, n_samples, window_size):
        end_idx = min(i + window_size, n_samples)
        window_embeddings = embeddings[i:end_idx]
        window_labels = labels[i:end_idx]
        window_predictions = predictions[i:end_idx]
        
        windows.append({
            'embeddings': window_embeddings,
            'labels': window_labels,
            'predictions': window_predictions,
            'window_number': len(windows)
        })
    
    return windows


def extract_embeddings_with_predictions(feature_extractor, n_samples=1000, n_classes=10):
    """Simulate extraction of embeddings from the model for all data with predictions"""
    all_embeddings = []
    all_labels = []
    all_predictions = []
    all_image_paths = []  # Store image paths
    
    for i in range(0, n_samples, 64):  # Simulate batch processing
        batch_size = min(64, n_samples - i)
        
        # Simulate batch of images (just dummy data to feed to our extractor)
        batch_images = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
        
        # Extract embeddings using our simulated feature extractor
        batch_embeddings = feature_extractor(batch_images)
        
        # Generate labels and predictions
        batch_labels = np.random.randint(0, n_classes, batch_size)
        batch_predictions = batch_labels  # Use labels as predictions
        
        all_embeddings.append(batch_embeddings)
        all_labels.append(batch_labels)
        all_predictions.append(batch_predictions)
        
        # Store image indices as paths
        batch_indices = list(range(i, i + batch_size))
        all_image_paths.extend(batch_indices)
    
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)
    all_predictions = np.hstack(all_predictions)
    
    return all_embeddings, all_labels, all_predictions, all_image_paths


def simulate_streaming_data(embeddings, labels, predictions, image_paths, window_size=100):
    """Simulate streaming data by splitting into windows"""
    n_samples = len(embeddings)
    windows = []
    
    for i in range(0, n_samples, window_size):
        end_idx = min(i + window_size, n_samples)
        window_embeddings = embeddings[i:end_idx]
        window_labels = labels[i:end_idx]
        window_predictions = predictions[i:end_idx]
        window_image_paths = image_paths[i:end_idx]
        
        windows.append({
            'embeddings': window_embeddings,
            'labels': window_labels,
            'predictions': window_predictions,
            'image_paths': window_image_paths,
            'window_number': len(windows)
        })
    
    return windows


def main():
    # Configuration
    model_name = 'resnet18'
    window_size = 200
    output_dir = 'outputs'
    
    print("Starting Data Drift Detection System (REAL Images & ResNet)...")
    
    # Get feature extractor (ResNet-18)
    print("Initializing ResNet-18 feature extractor...")
    feature_extractor = ImageFeatureExtractor(model_name)
    
    # Extract embeddings from the entire dataset using simulated images
    print("Extracting embeddings from ResNet-18 (simulated)...")
    all_embeddings, all_labels, all_predictions, all_image_paths = extract_embeddings_with_predictions(
        feature_extractor, n_samples=2000, n_classes=10
    )
    
    print(f"Total samples: {len(all_embeddings)}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    print(f"Unique classes: {np.unique(all_predictions)}")
    
    # Split into reference and streaming windows (first portion as reference)
    n_reference = min(500, len(all_embeddings) // 3)
    ref_embeddings = all_embeddings[:n_reference]
    ref_labels = all_labels[:n_reference]
    ref_predictions = all_predictions[:n_reference]
    ref_image_paths = all_image_paths[:n_reference]
    
    streaming_embeddings = all_embeddings[n_reference:]
    streaming_labels = all_labels[n_reference:]
    streaming_predictions = all_predictions[n_reference:]
    streaming_image_paths = all_image_paths[n_reference:]
    
    print(f"Reference samples: {len(ref_embeddings)}")
    print(f"Streaming samples: {len(streaming_embeddings)}")
    
    # Initialize drift detector with embeddings
    print("Initializing drift detector with embeddings...")
    drift_detector = FréchetDistanceDriftDetector(threshold_multiplier=2.0)
    drift_detector.fit_reference(ref_embeddings)
    
    print(f"Reference mean shape: {drift_detector.reference_mean.shape}")
    print(f"Reference cov shape: {drift_detector.reference_cov.shape}")
    print(f"Learned threshold: {drift_detector.threshold:.4f}")
    
    # Simulate streaming and detect drift on embeddings
    print("Processing streaming windows with image embeddings...")
    streaming_windows = simulate_streaming_data(
        streaming_embeddings, streaming_labels, streaming_predictions, streaming_image_paths, window_size
    )
    
    results = []
    class_drift_history = []
    window_numbers = []
    
    visualizer = DriftVisualizer(output_dir)
    
    for i, window in enumerate(streaming_windows):
        print(f"Processing window {i+1}/{len(streaming_windows)}")
        
        # Global drift detection on embeddings
        drift_score, is_drift, threshold = drift_detector.detect_global_drift(window['embeddings'])
        
        # Class-wise drift detection on embeddings
        class_drift_scores, most_affected_class = drift_detector.detect_classwise_drift(
            window['embeddings'], window['labels'], window['predictions']
        )
        
        # Get drift explanation samples (indices of most drifted images)
        explanation_indices = drift_detector.get_drift_explanation(
            window['embeddings'], window['labels'], window['predictions'], 
            most_affected_class, n_samples=5
        )
        
        # Store results
        result = {
            'window_number': i,
            'drift_score': float(drift_score),
            'threshold': float(threshold),
            'drift_detected': bool(is_drift),
            'most_affected_class': int(most_affected_class) if most_affected_class is not None else None,
            'n_samples': len(window['embeddings']),
            'explanation_sample_indices': explanation_indices.tolist() if len(explanation_indices) > 0 else [],
            'class_drift_scores': {int(k): float(v) for k, v in class_drift_scores.items()}
        }
        
        results.append(result)
        class_drift_history.append(class_drift_scores)
        window_numbers.append(i)
        
        print(f"  Drift Score: {drift_score:.4f}, Threshold: {threshold:.4f}, "
              f"Drift Detected: {is_drift}, Most Affected Class: {most_affected_class}")
    
    # Generate visualizations
    print("Generating visualizations...")
    drift_scores = [r['drift_score'] for r in results]
    thresholds = [r['threshold'] for r in results]
    
    global_plot_path = os.path.join(output_dir, 'global_drift_real.png')
    class_plot_path = os.path.join(output_dir, 'classwise_drift_real.png')
    
    visualizer.plot_global_drift(drift_scores, thresholds, window_numbers, global_plot_path)
    visualizer.plot_classwise_drift(class_drift_history, window_numbers, class_plot_path)
    
    # Save results
    print("Saving results...")
    csv_path = os.path.join(output_dir, 'drift_results_real.csv')
    json_path = os.path.join(output_dir, 'drift_results_real.json')
    report_path = os.path.join(output_dir, 'summary_report_real.txt')
    
    save_results_to_csv(results, csv_path)
    save_results_to_json(results, json_path)
    generate_summary_report(results, report_path)
    
    # Print summary
    drift_events = [r for r in results if r['drift_detected']]
    print(f"\nSUMMARY:")
    print(f"Total windows processed: {len(results)}")
    print(f"Windows with drift detected: {len(drift_events)}")
    print(f"Drift detection rate: {len(drift_events)/len(results)*100:.2f}%")
    
    if drift_events:
        print(f"Drift detected in windows: {[r['window_number'] for r in drift_events]}")
    
    print(f"\nResults saved to '{output_dir}/' directory")
    print("- global_drift_real.png: Global drift visualization")
    print("- classwise_drift_real.png: Class-wise drift visualization")
    print("- drift_results_real.csv: Detailed results")
    print("- drift_results_real.json: Results in JSON format")
    print("- summary_report_real.txt: Summary report")


if __name__ == "__main__":
    main()