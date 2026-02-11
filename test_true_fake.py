import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from image_dataset import ImageFolderDatasetWithBLIP
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from DFAD_model_base import DFADModel
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, model):
    """Load model weights from checkpoint"""
    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # Handle DataParallel weights if present
    if list(model_state.keys())[0].startswith('module.'):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    model.load_state_dict(model_state)
    print("✓ Model loaded successfully")
    
    return model

def test_model(model, test_loader, model_name="Model", threshold=0.5):
    """Test model and compute comprehensive metrics"""
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    print(f"\nRunning inference for {model_name}...")
    with torch.no_grad():
        for img_features, text_features, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            img_features = img_features.to(device).float()
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            
            output = model(img_features, text_features).squeeze()
            probabilities = torch.sigmoid(output)
            predictions = (probabilities > threshold).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
    metrics['precision'] = precision_score(all_labels, all_predictions, zero_division=0)
    metrics['recall'] = recall_score(all_labels, all_predictions, zero_division=0)
    metrics['f1_score'] = f1_score(all_labels, all_predictions, zero_division=0)
    
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probabilities)
    else:
        metrics['auc'] = None
    
    cm = confusion_matrix(all_labels, all_predictions)
    metrics['confusion_matrix'] = cm
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics['num_real'] = int(np.sum(all_labels == 0))
    metrics['num_fake'] = int(np.sum(all_labels == 1))
    metrics['total_samples'] = len(all_labels)
    
    return metrics, all_labels, all_predictions, all_probabilities

def compare_models(
    data_dir,
    model1_checkpoint,
    model2_checkpoint,
    model1_name="Model 1",
    model2_name="Model 2",
    batch_size=128,
    use_blip=True,
    threshold=0.5,
    output_dir='model_comparison',
    num_workers=0
):
    """
    Compare two models on the same test dataset
    
    Args:
        data_dir: Directory containing test images
        model1_checkpoint: Path to first model checkpoint
        model2_checkpoint: Path to second model checkpoint
        model1_name: Display name for first model
        model2_name: Display name for second model
        batch_size: Batch size for testing
        use_blip: Whether to use BLIP for caption generation
        threshold: Classification threshold
        output_dir: Directory to save comparison results
        num_workers: Number of workers for data loading
    """
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"Test data directory: {data_dir}")
    print(f"{model1_name}: {model1_checkpoint}")
    print(f"{model2_name}: {model2_checkpoint}")
    print(f"Use BLIP captioning: {use_blip}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test dataset with BLIP (shared for both models)
    print("Loading test dataset with BLIP caption generation...")
    test_dataset = ImageFolderDatasetWithBLIP(
        root_dir=data_dir,
        use_blip=use_blip,
        cache_captions=True
    )
    
    print(f"Test dataset size: {len(test_dataset)} images\n")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Load Model 1
    print(f"\n{'='*80}")
    print(f"Loading {model1_name}")
    print(f"{'='*80}")
    model1 = DFADModel()
    model1 = load_model(model1_checkpoint, model1)
    model1 = model1.to(device)
    
    # Load Model 2
    print(f"\n{'='*80}")
    print(f"Loading {model2_name}")
    print(f"{'='*80}")
    model2 = DFADModel()
    model2 = load_model(model2_checkpoint, model2)
    model2 = model2.to(device)
    
    # Test Model 1
    print(f"\n{'='*80}")
    print(f"Testing {model1_name}")
    print(f"{'='*80}")
    metrics1, labels1, preds1, probs1 = test_model(model1, test_loader, model1_name, threshold)
    
    # Test Model 2
    print(f"\n{'='*80}")
    print(f"Testing {model2_name}")
    print(f"{'='*80}")
    metrics2, labels2, preds2, probs2 = test_model(model2, test_loader, model2_name, threshold)
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<20} {model1_name:>15} {model2_name:>15} {'Difference':>15}")
    print(f"{'-'*70}")
    
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity']
    
    for metric in comparison_metrics:
        if metric in metrics1 and metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                winner = "<M2" if diff > 0 else "<M1" if diff < 0 else "TIE"
                print(f"{metric:<20} {val1:>15.6f} {val2:>15.6f} {diff:>+14.6f} {winner}")
    
    print(f"\n{'='*80}\n")
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, 'comparison_results.txt')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{model1_name}: {model1_checkpoint}\n")
        f.write(f"{model2_name}: {model2_checkpoint}\n\n")
        
        f.write(f"{'Metric':<20} {model1_name:>15} {model2_name:>15} {'Difference':>15}\n")
        f.write("-"*70 + "\n")
        
        for metric in comparison_metrics:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                if val1 is not None and val2 is not None:
                    diff = val2 - val1
                    winner = "<M2" if diff > 0 else "<M1" if diff < 0 else "TIE"
                    f.write(f"{metric:<20} {val1:>15.6f} {val2:>15.6f} {diff:>+14.6f} {winner}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed metrics for each model
        for model_name, metrics in [(model1_name, metrics1), (model2_name, metrics2)]:
            f.write(f"\n{model_name} - Detailed Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Accuracy:     {metrics['accuracy']:.6f}\n")
            f.write(f"  Precision:    {metrics['precision']:.6f}\n")
            f.write(f"  Recall:       {metrics['recall']:.6f}\n")
            f.write(f"  F1-Score:     {metrics['f1_score']:.6f}\n")
            if metrics['auc'] is not None:
                f.write(f"  AUC-ROC:      {metrics['auc']:.6f}\n")
            if 'specificity' in metrics:
                f.write(f"  Specificity:  {metrics['specificity']:.6f}\n")
            
            if metrics['confusion_matrix'].shape == (2, 2):
                f.write(f"\n  Confusion Matrix:\n")
                f.write(f"                Predicted\n")
                f.write(f"              Real    Fake\n")
                f.write(f"  Actual Real  {metrics['true_negatives']:4d}    {metrics['false_positives']:4d}\n")
                f.write(f"  Actual Fake  {metrics['false_negatives']:4d}    {metrics['true_positives']:4d}\n")
            f.write("\n")
    
    print(f"✓ Comparison results saved to: {comparison_file}")
    
    # Create comparison visualization
    try:
        create_comparison_plot(metrics1, metrics2, model1_name, model2_name, output_dir)
    except Exception as e:
        print(f"Warning: Could not create comparison plot: {e}")
    
    # Save individual predictions
    np.savez(
        os.path.join(output_dir, f'{model1_name.replace(" ", "_")}_predictions.npz'),
        labels=labels1,
        predictions=preds1,
        probabilities=probs1
    )
    
    np.savez(
        os.path.join(output_dir, f'{model2_name.replace(" ", "_")}_predictions.npz'),
        labels=labels2,
        predictions=preds2,
        probabilities=probs2
    )
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return metrics1, metrics2

def create_comparison_plot(metrics1, metrics2, model1_name, model2_name, output_dir):
    """Create bar chart comparing model metrics"""
    
    metrics_to_plot = []
    model1_values = []
    model2_values = []
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity']:
        if metric in metrics1 and metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            if val1 is not None and val2 is not None:
                metrics_to_plot.append(metric.replace('_', ' ').title())
                model1_values.append(val1)
                model2_values.append(val2)
    
    if not metrics_to_plot:
        return
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, model1_values, width, label=model1_name, alpha=0.8)
    bars2 = ax.bar(x + width/2, model2_values, width, label=model2_name, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to: {plot_path}")

if __name__ == '__main__':
    # Configuration
    TEST_DATA_DIR = r'C:\Users\bacal\Documents\Testing'
    
    # MODEL1_CHECKPOINT = r'relu_af_checkpoints_auc_gamma_0.8_approx/checkpoint_epoch_0031.pt'
    MODEL1_CHECKPOINT = r'relu_checkpoints_from_images_with_blip/checkpoint_epoch_0049.pt'
    MODEL1_NAME = "RELU"
    
    # MODEL2_CHECKPOINT = r'gelu_af_checkpoints_auc_gamma_0.8_approx/checkpoint_epoch_0031.pt'
    MODEL2_CHECKPOINT = r'gelu_checkpoints_from_images_with_blip/checkpoint_epoch_0049.pt'
    MODEL2_NAME = "GELU"
    
    # Run comparison
    compare_models(
        data_dir=TEST_DATA_DIR,
        model1_checkpoint=MODEL1_CHECKPOINT,
        model2_checkpoint=MODEL2_CHECKPOINT,
        model1_name=MODEL1_NAME,
        model2_name=MODEL2_NAME,
        batch_size=64,
        use_blip=True,
        threshold=0.5,
        output_dir='model_comparison',
        num_workers=0
    )