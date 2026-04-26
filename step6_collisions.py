"""
Step 6: Feature Collisions as a Function of Feature Dimension

This script tests the paper's claim that feature collisions become unavoidable
when feature dimension m is too low, especially under neural collapse conditions.

Key experiments:
- 6.1: Measure collision mass C = fraction of secondary classes per Voronoi cell
- 6.2: Vary feature dimension m = 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16
- 6.3: Identify critical dimension where collisions drop below threshold

Author: Based on "Deep Networks Learn Voronoi Partitions of Feature Space"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
from tqdm import tqdm
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. MODEL ARCHITECTURE (Configurable Feature Dimension)
# ============================================================================

class VoronoiCNN(nn.Module):
    def __init__(self, input_channels=3, feature_dim=2, num_classes=10):
        super(VoronoiCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.feature_layer = nn.Linear(256, feature_dim)
        
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        feats = self.feature_extractor(x)
        feats = self.flatten(feats)
        feats = self.fc1(feats)
        feats = self.bn1(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        feats = self.fc2(feats)
        feats = self.bn2(feats)
        feats = self.relu(feats)
        features = self.feature_layer(feats)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


# ============================================================================
# 2. VORONOI UTILITIES
# ============================================================================

def assign_voronoi_cells(features, prototypes, device=None):
    """Assign each feature vector to its nearest prototype (Voronoi cell)"""
    if device is None:
        device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    distances = torch.cdist(features, prototypes)
    cell_assignments = torch.argmin(distances, dim=1)
    return cell_assignments


def compute_collision_mass(features, labels, prototypes):
    """
    Compute collision mass per cell (Definition 2 in paper):
    C_i = ∫_{V_i} Σ_{y≠y_i} dν^{(y)}(φ)
    
    Returns:
        collision_per_cell: dict cell_id -> collision mass (0 to 1)
        overall_collision: float (average across cells)
        total_collision_frac: float (total secondary samples / total samples)
    """
    device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    labels = labels.to(device)
    
    cell_assignments = assign_voronoi_cells(features, prototypes, device=device)
    
    k = len(prototypes)
    collision_per_cell = {}
    total_collision = 0
    total_samples = 0
    
    for i in range(k):
        mask = (cell_assignments == i)
        if mask.sum() == 0:
            collision_per_cell[i] = 0.0
            continue
        
        labels_in_cell = labels[mask]
        # Primary class for this cell is i (by prototype assignment)
        correct_count = (labels_in_cell == i).sum().item()
        total_in_cell = mask.sum().item()
        collision = 1 - (correct_count / total_in_cell)
        collision_per_cell[i] = collision
        total_collision += total_in_cell - correct_count
        total_samples += total_in_cell
    
    overall_collision = np.mean(list(collision_per_cell.values())) if collision_per_cell else 0
    total_collision_frac = total_collision / total_samples if total_samples > 0 else 0
    
    return collision_per_cell, overall_collision, total_collision_frac


def compute_separation_quality(features, labels, prototypes):
    """
    Compute additional metrics for feature separation quality.
    
    Returns:
        intra_class_dist: mean distance within class to prototype
        inter_class_dist: mean distance between class prototypes
        separation_ratio: inter / intra (higher = better)
    """
    device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    labels = labels.to(device)
    
    k = len(prototypes)
    
    # Intra-class distances (within class to prototype)
    intra_dists = []
    for i in range(k):
        mask = (labels == i)
        if mask.sum() > 0:
            class_features = features[mask]
            dists = torch.norm(class_features - prototypes[i], dim=1)
            intra_dists.append(dists.mean().item())
    intra_class_dist = np.mean(intra_dists) if intra_dists else 0
    
    # Inter-class distances (between prototypes)
    inter_dists = []
    for i in range(k):
        for j in range(i+1, k):
            dist = torch.norm(prototypes[i] - prototypes[j]).item()
            inter_dists.append(dist)
    inter_class_dist = np.mean(inter_dists) if inter_dists else 0
    
    separation_ratio = inter_class_dist / (intra_class_dist + 1e-8)
    
    return intra_class_dist, inter_class_dist, separation_ratio


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device, compute_features=False):
    model.eval()
    correct = 0
    total = 0
    all_features = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        
        if compute_features:
            _, features = model.forward(images, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
        
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    
    if compute_features:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return accuracy, all_features, all_labels
    
    return accuracy


def compute_prototypes(model, dataloader, device):
    """Compute class prototypes as mean features per class"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing prototypes", leave=False):
            images = images.to(device)
            _, features = model.forward(images, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    prototypes = []
    for c in range(model.classifier.out_features):
        mask = all_labels == c
        if mask.sum() > 0:
            proto = all_features[mask].mean(dim=0)
        else:
            proto = torch.zeros(model.feature_layer.out_features)
        prototypes.append(proto)
    
    return torch.stack(prototypes).to(device)


# ============================================================================
# 4. SINGLE DIMENSION EXPERIMENT
# ============================================================================

def run_single_dim_experiment(feature_dim, epochs=80, use_full_dataset=False):
    """
    Run experiment for a single feature dimension.
    Returns dict of results.
    """
    
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dataset': 'CIFAR-10',
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Testing feature dimension: m = {feature_dim}")
    print(f"{'='*60}")
    
    # Load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    full_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    if use_full_dataset:
        train_dataset = full_train
        test_dataset = full_test
    else:
        train_dataset = Subset(full_train, range(8000))
        test_dataset = Subset(full_test, range(2000))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = VoronoiCNN(input_channels=3, feature_dim=feature_dim, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_accuracy = 0
    final_collision = 0
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == config['epochs']:
            test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
            prototypes = compute_prototypes(model, train_loader, device)
            
            _, overall_collision, total_collision_frac = compute_collision_mass(features, labels, prototypes)
            intra, inter, sep_ratio = compute_separation_quality(features, labels, prototypes)
            
            print(f"  Epoch {epoch}: Acc={test_acc:.4f}, Collision={overall_collision:.4f}, Sep={sep_ratio:.2f}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                final_collision = overall_collision
    
    # Final evaluation
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = compute_prototypes(model, train_loader, device)
    collision_per_cell, overall_collision, total_collision_frac = compute_collision_mass(features, labels, prototypes)
    intra_class_dist, inter_class_dist, separation_ratio = compute_separation_quality(features, labels, prototypes)
    
    results = {
        'feature_dim': feature_dim,
        'test_accuracy': test_acc,
        'best_accuracy': best_accuracy,
        'collision_mass': overall_collision,
        'total_collision_frac': total_collision_frac,
        'collision_per_cell': collision_per_cell,
        'intra_class_dist': intra_class_dist,
        'inter_class_dist': inter_class_dist,
        'separation_ratio': separation_ratio,
        'epochs_trained': config['epochs']
    }
    
    print(f"\n  FINAL: m={feature_dim}, Acc={test_acc:.4f}, Collision={overall_collision:.4f}")
    
    return results


# ============================================================================
# 5. MAIN EXPERIMENT (STEP 6) - SWEEP OVER DIMENSIONS
# ============================================================================

def run_step6_experiment(dimensions=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16], 
                         epochs=80, 
                         use_full_dataset=False):
    """
    Step 6: Feature Collisions as a Function of Feature Dimension
    
    Sweeps over multiple feature dimensions and measures collision mass.
    Identifies critical dimension where collisions drop below threshold.
    """
    
    save_dir = f'step6_results_collisions_sweep'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Step 6: Feature Collisions as a Function of Feature Dimension")
    print("="*70)
    print(f"Testing dimensions: {dimensions}")
    print(f"Epochs per dimension: {epochs}")
    print(f"k = 10 classes")
    print(f"Critical threshold: m = k - 1 = 9 (from Theorem 7)")
    print("="*70 + "\n")
    
    all_results = []
    
    for dim in dimensions:
        start_time = time.time()
        results = run_single_dim_experiment(dim, epochs, use_full_dataset)
        results['time_minutes'] = (time.time() - start_time) / 60
        all_results.append(results)
        
        # Save intermediate results
        with open(f'{save_dir}/results_dim{dim}.json', 'w') as f:
            # Convert numpy types to Python types
            clean_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    clean_results[k] = {str(kk): float(vv) for kk, vv in v.items()}
                elif hasattr(v, 'tolist'):
                    clean_results[k] = v.tolist()
                else:
                    clean_results[k] = float(v) if isinstance(v, (np.float32, np.float64)) else v
            json.dump(clean_results, f, indent=2)
    
    # Aggregate results
    dims = [r['feature_dim'] for r in all_results]
    accuracies = [r['test_accuracy'] for r in all_results]
    collisions = [r['collision_mass'] for r in all_results]
    total_collisions = [r['total_collision_frac'] for r in all_results]
    separation_ratios = [r['separation_ratio'] for r in all_results]
    
    # Find critical dimension (collision < 0.1 threshold)
    critical_dim = None
    for i, coll in enumerate(collisions):
        if coll < 0.1:
            critical_dim = dims[i]
            break
    
    print("\n" + "="*70)
    print("Step 6 Results Summary")
    print("="*70)
    print(f"{'m (dim)':<10} {'Accuracy':<12} {'Collision Mass':<15} {'Separation Ratio':<15}")
    print("-" * 55)
    for i, dim in enumerate(dims):
        print(f"{dim:<10} {accuracies[i]:<12.4f} {collisions[i]:<15.4f} {separation_ratios[i]:<15.2f}")
    print("-" * 55)
    
    print(f"\nTheorem 7 predicts: m ≥ 9 for collision-free configuration")
    print(f"Observed critical dimension (collision < 0.1): {critical_dim if critical_dim else '>16'}")
    
    # Generate plots
    plot_step6_results(dims, accuracies, collisions, total_collisions, separation_ratios, 
                      critical_dim, save_dir)
    
    # Save all results
    final_results = {
        'dimensions': dims,
        'accuracies': accuracies,
        'collision_masses': collisions,
        'total_collision_fractions': total_collisions,
        'separation_ratios': separation_ratios,
        'critical_dimension': critical_dim,
        'theorem7_prediction': 9,  # k - 1
        'per_dim_results': all_results
    }
    
    torch.save(final_results, f'{save_dir}/all_results.pt')
    
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump({k: v for k, v in final_results.items() if k != 'per_dim_results'}, f, indent=2)
    
    print(f"\nResults saved to {save_dir}/")
    print("Step 6 Complete!")
    
    return final_results


# ============================================================================
# 6. PLOTTING FUNCTIONS
# ============================================================================

def plot_step6_results(dims, accuracies, collisions, total_collisions, separation_ratios, 
                       critical_dim, save_dir):
    """Create comprehensive figures for Step 6"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Collision Mass vs Feature Dimension (KEY FIGURE)
    axes[0, 0].plot(dims, collisions, 'bo-', linewidth=2, markersize=8, label='Collision Mass')
    axes[0, 0].fill_between(dims, 0, collisions, alpha=0.3)
    axes[0, 0].axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Threshold (10% collision)')
    axes[0, 0].axvline(x=9, color='green', linestyle='--', linewidth=2, label='Theorem 7: m = k-1 = 9')
    if critical_dim:
        axes[0, 0].axvline(x=critical_dim, color='purple', linestyle=':', linewidth=2, 
                          label=f'Critical dim = {critical_dim}')
    axes[0, 0].set_xlabel('Feature Dimension (m)')
    axes[0, 0].set_ylabel('Collision Mass (C)')
    axes[0, 0].set_title('Step 6.1-6.2: Collision Mass vs Feature Dimension')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Plot 2: Test Accuracy vs Feature Dimension
    axes[0, 1].plot(dims, accuracies, 'gs-', linewidth=2, markersize=8)
    axes[0, 1].fill_between(dims, 0, accuracies, alpha=0.3)
    axes[0, 1].axvline(x=9, color='green', linestyle='--', linewidth=2, label='Theorem 7: m = k-1 = 9')
    axes[0, 1].set_xlabel('Feature Dimension (m)')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Classification Accuracy vs Feature Dimension')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Plot 3: Separation Ratio vs Feature Dimension
    axes[1, 0].plot(dims, separation_ratios, 'purple', marker='D', linewidth=2, markersize=8)
    axes[1, 0].fill_between(dims, 0, separation_ratios, alpha=0.3)
    axes[1, 0].axvline(x=9, color='green', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Feature Dimension (m)')
    axes[1, 0].set_ylabel('Separation Ratio (inter/intra)')
    axes[1, 0].set_title('Class Separation Quality vs Feature Dimension')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Collision Mass vs Accuracy (correlation)
    axes[1, 1].scatter(collisions, accuracies, s=100, c=dims, cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Collision Mass (C)')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Collision-Accuracy Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Feature Dimension (m)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/step6_summary.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/step6_summary.png")
    
    # Also create a detailed bar chart for collisions per dimension
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars = ax2.bar([str(d) for d in dims], collisions, color='steelblue', alpha=0.7)
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='10% collision threshold')
    ax2.set_xlabel('Feature Dimension (m)')
    ax2.set_ylabel('Collision Mass')
    ax2.set_title('Step 6: Collision Mass by Feature Dimension (k=10 classes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, coll in zip(bars, collisions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{coll:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/collision_barchart.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/collision_barchart.png")


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 6: Feature Collisions vs Dimension')
    parser.add_argument('--dims', type=str, default='2,3,4,5,6,7,8,9,10,12,16',
                        help='Comma-separated list of dimensions to test')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Epochs per dimension')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer dimensions for testing')
    
    args = parser.parse_args()
    
    if args.quick:
        dims = [2, 4, 6, 8, 10, 12]
        epochs = 40
        print("Quick mode enabled")
    else:
        dims = [int(d) for d in args.dims.split(',')]
        epochs = args.epochs
    
    results = run_step6_experiment(
        dimensions=dims,
        epochs=epochs,
        use_full_dataset=args.full
    )
