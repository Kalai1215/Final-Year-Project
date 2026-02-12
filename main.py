import os
import numpy as np
import torch
from data_drift_project.data_loader import load_image_data, get_dataloader
from data_drift_project.model_extractor import get_feature_extractor, extract_embeddings
from data_drift_project.drift_detector import FréchetDistanceDriftDetector
from data_drift_project.visualizer import DriftVisualizer, save_results_to_csv, generate_summary_report

def main():
    # Configuration (As per PPT stack)
    window_size = 200
    output_dir = 'outputs'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting DRIFTLENS... Device: {device}")
    
    # 1. Data Loading
    dataset = load_image_data(data_type='cifar10')
    dataloader = get_dataloader(dataset, batch_size=64, shuffle=False)
    
    # 2. Feature Extraction
    feature_extractor = get_feature_extractor('image', 'resnet18').to(device)
    all_embeddings, all_labels, all_predictions = extract_embeddings(feature_extractor, dataloader, device)
    
    # 3. Offline Phase: Reference Distribution & Threshold
    n_ref = 500
    ref_embeddings = all_embeddings[:n_ref]
    
    drift_detector = FréchetDistanceDriftDetector(threshold_multiplier=2.5, n_components=10)
    drift_detector.fit_reference(ref_embeddings)
    print(f"Offline Phase Complete. Threshold: {drift_detector.threshold:.4f}")
    
    # 4. Online Phase: Streaming Drift Detection
    streaming_embeddings = all_embeddings[n_ref:]
    streaming_preds = all_predictions[n_ref:]
    
    results = []
    visualizer = DriftVisualizer(output_dir)
    
    for i in range(0, len(streaming_embeddings), window_size):
        end = i + window_size
        win_embeddings = streaming_embeddings[i:end]
        win_preds = streaming_preds[i:end]
        win_num = i // window_size
        
        # Detection
        score, is_drift, thresh = drift_detector.detect_global_drift(win_embeddings)
        class_scores, affected_cls = drift_detector.detect_classwise_drift(win_embeddings, win_preds)
        
        # PENDING WORK: Explanation (Prototypes)
        prototypes = []
        if is_drift:
            prototypes = drift_detector.get_drift_explanation_prototypes(win_embeddings, affected_cls, win_preds)
        
        results.append({
            'window_number': win_num,
            'drift_score': float(score),
            'threshold': float(thresh),
            'drift_detected': bool(is_drift),
            'most_affected_class': int(affected_cls) if affected_cls is not None else -1,
            'prototypes_indices': prototypes
        })
        
        print(f"Window {win_num}: Score={score:.3f} | Drift={is_drift} | Affected Class={affected_cls}")

    # 5. Visualization and Reporting
    save_results_to_csv(results, os.path.join(output_dir, 'drift_results.csv'))
    generate_summary_report(results, os.path.join(output_dir, 'summary.txt'))
    
    # Plotting
    scores = [r['drift_score'] for r in results]
    threshs = [r['threshold'] for r in results]
    visualizer.plot_global_drift(scores, threshs, list(range(len(results))), os.path.join(output_dir, 'drift_plot.png'))

    print(f"Project Implementation Complete. Results in '{output_dir}'")

if __name__ == "__main__":
    main()