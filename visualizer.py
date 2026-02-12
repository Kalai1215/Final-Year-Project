import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


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