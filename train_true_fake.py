import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from image_dataset import ImageFolderDatasetWithBLIP
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from DFAD_model_base import DFADModel
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, optimizer, criterion, train_loader):
    """Simple training loop"""
    model.train()
    total_loss = 0
    
    for img_features, text_features, labels in tqdm(train_loader, desc="Training"):
        img_features = img_features.to(device).float()
        text_features = text_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(img_features, text_features).squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader):
    """Evaluate model"""
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for img_features, text_features, labels in tqdm(val_loader, desc="Evaluating"):
            img_features = img_features.to(device).float()
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            output = model(img_features, text_features).squeeze()
            probabilities = torch.sigmoid(output)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(probabilities.cpu().numpy())
    
    import numpy as np
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    accuracy = ((all_predictions > 0.5) == all_labels).mean()
    auc = roc_auc_score(all_labels, all_predictions)
    
    return accuracy, auc

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return starting epoch and best metrics"""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Format: {'epoch': ..., 'model_state_dict': ..., 'optimizer_state_dict': ...}
        print("  Detected: Full checkpoint format")
        model_state = checkpoint['model_state_dict']
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get epoch and metrics
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_auc = checkpoint.get('val_auc', 0.0)
        best_acc = checkpoint.get('val_accuracy', 0.0)
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Previous Val AUC: {checkpoint.get('val_auc', 'N/A')}")
        print(f"  Previous Val Accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        print(f"  Previous Train Loss: {checkpoint.get('train_loss', 'N/A')}\n")
        
    else:
        # Format: Direct model state dict (weights only)
        print("  Detected: Weights-only format")
        model_state = checkpoint
        start_epoch = 0
        best_auc = 0.0
        best_acc = 0.0
        
        print(f"✓ Model weights loaded successfully")
        print(f"  Note: No training state found (epoch, optimizer, scheduler)")
        print(f"  Starting from epoch 0 with loaded weights\n")
    
    # Handle DataParallel weights if present
    if list(model_state.keys())[0].startswith('module.'):
        print("  Removing 'module.' prefix from DataParallel weights...")
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    # Load model state
    model.load_state_dict(model_state)
    
    return start_epoch, best_auc, best_acc

def train_from_images(
    data_dir,
    batch_size=128,
    num_epochs=50,
    learning_rate=1e-3,
    train_split=0.8,
    use_blip=True,
    resume_from=None,
    use_data_parallel=False,  # NEW: Enable DataParallel for multi-GPU
    num_workers=0  # NEW: Configurable num_workers
):
    """Train model from image folder with BLIP captioning"""
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Path does not exist: {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"TRAINING WITH CUSTOM DFAD MODEL")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Use BLIP captioning: {use_blip}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Train/Val split: {train_split:.0%}/{1-train_split:.0%}")
    print(f"Resume from checkpoint: {resume_from if resume_from else 'None (training from scratch)'}")
    print(f"DataParallel: {use_data_parallel}")
    print(f"Num workers: {num_workers}")
    print(f"Device: {device}\n")
    
    # Load dataset with BLIP
    print("Loading dataset with BLIP caption generation...")
    dataset = ImageFolderDatasetWithBLIP(
        root_dir=data_dir, 
        use_blip=use_blip,
        cache_captions=True
    )
    
    if len(dataset) == 0:
        print("ERROR: No images found in dataset!")
        return
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split:")
    print(f"  Training: {train_size} images")
    print(f"  Validation: {val_size} images\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Initialize YOUR custom model
    print("Initializing CUSTOM DFAD model...")
    model = DFADModel()
    
    # Apply DataParallel if requested (for multi-GPU training)
    if use_data_parallel and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using DataParallel with {gpu_count} GPUs: {list(range(gpu_count))}")
            model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
        else:
            print(f"DataParallel requested but only 1 GPU available, using single GPU")
    
    model = model.to(device)
    print(f"Model architecture:\n{model}\n")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Setup saving
    output_dir = 'checkpoints_custom_dfad'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize training state
    start_epoch = 0
    best_auc = 0.0
    best_acc = 0.0
    
    # Load checkpoint if resuming
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_auc, best_acc = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
    elif resume_from:
        print(f"WARNING: Checkpoint not found at {resume_from}")
        print("Starting training from scratch...\n")
    
    # Setup metrics file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    
    # Create or append to metrics file
    if start_epoch == 0:
        with open(metrics_file, 'w') as f:
            f.write('Epoch,Train Loss,Val Accuracy,Val AUC,Learning Rate\n')
    else:
        print(f"Appending to existing metrics file: {metrics_file}\n")
    
    # Save some sample captions (only if starting fresh)
    if start_epoch == 0:
        caption_samples_file = os.path.join(output_dir, 'sample_captions.txt')
        with open(caption_samples_file, 'w', encoding='utf-8') as f:
            f.write('Sample Generated Captions:\n')
            f.write('='*60 + '\n\n')
            for i, (filename, caption) in enumerate(list(dataset.caption_cache.items())[:20]):
                f.write(f"{filename}: {caption}\n")
        print(f"Sample captions saved to: {caption_samples_file}\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs-1}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        accuracy, auc = evaluate(model, val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nResults:")
        print(f"  Train Loss:    {train_loss:.6f}")
        print(f"  Val Accuracy:  {accuracy:.6f}")
        print(f"  Val AUC:       {auc:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save metrics
        with open(metrics_file, 'a') as f:
            f.write(f'{epoch},{train_loss},{accuracy},{auc},{current_lr}\n')
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch:04d}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_accuracy': accuracy,
            'val_auc': auc,
        }, checkpoint_path)
        
        # Save best models
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_auc.pt'))
            print(f"  🌟 New best AUC: {best_auc:.6f}")
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_acc.pt'))
            print(f"  🌟 New best Accuracy: {best_acc:.6f}")
        
        scheduler.step()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation AUC: {best_auc:.6f}")
    print(f"Best Validation Accuracy: {best_acc:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    # Windows path
    DATA_DIR = r'C:\Users\bacal\Documents\Testing'
    
    # To resume from a checkpoint, set the path here:
    RESUME_CHECKPOINT = r'gelu_af_checkpoints_auc_gamma_0.5_new/checkpoint_epoch_0031.pt'  # Set to checkpoint path to resume
    
    train_from_images(
        data_dir=DATA_DIR,
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-3,
        train_split=0.8,
        use_blip=True,
        resume_from=RESUME_CHECKPOINT,
        use_data_parallel=False,  # Set to True for multi-GPU training
        num_workers=0  # Set to 8 for Linux, 0 for Windows
    )