"""
Step 7: Symmetry vs Separation - Neural Collapse (Theorem 7)

This script tests Theorem 7 which states:
When class prototypes are required to be pairwise equidistant with equal norms
and zero mean (neural collapse regime), the feature dimension must satisfy m >= k-1.

Experiments:
- 7.1: Compare standard training vs neural collapse regime (with weight decay + normalization)
- 7.2: Test dimensionality bound: collisions become unavoidable when m < k-1 under collapse
- 7.3: Measure how close the network gets to the simplex ETF configuration

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
# 1. MODEL ARCHITECTURE
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
# 2. NEURAL COLLAPSE METRICS
# ============================================================================

def compute_prototypes_from_features(features, labels, num_classes):
    """Compute class prototypes as mean features per class"""
    device = features.device
    prototypes = torch.zeros(num_classes, features.shape[1], device=device)
    counts = torch.zeros(num_classes, device=device)
    
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = features[mask].mean(dim=0)
            counts[c] = mask.sum().float()
    
    return prototypes, counts


def compute_neural_collapse_metrics(prototypes):
    """
    Compute metrics for neural collapse (Theorem 7):
    1. Within-class variability collapse: should be small
    2. Between-class separation: should be maximized
    3. Equiangular condition: pairwise angles should be equal
    4. ETF similarity: <zi, zj> = -||z1||^2 / (k-1)
    
    Returns:
        metrics: dict with various collapse metrics
    """
    k = len(prototypes)
    prototypes = prototypes.detach().cpu()
    
    # Center the prototypes (zero mean)
    centered = prototypes - prototypes.mean(dim=0, keepdim=True)
    
    # Compute norms
    norms = torch.norm(prototypes, dim=1)
    norm_mean = norms.mean().item()
    norm_std = norms.std().item()
    norm_ratio = norm_std / (norm_mean + 1e-8)
    
    # Compute pairwise cosine similarities
    similarities = []
    for i in range(k):
        for j in range(i+1, k):
            cos_sim = torch.dot(prototypes[i], prototypes[j]) / (norms[i] * norms[j] + 1e-8)
            similarities.append(cos_sim.item())
    
    sim_mean = np.mean(similarities)
    sim_std = np.std(similarities)
    
    # Theoretical ETF similarity for ideal collapse: -1/(k-1)
    theoretical_sim = -1.0 / (k - 1)
    
    # Compute deviation from ETF
    etf_deviation = abs(sim_mean - theoretical_sim)
    
    # Compute pairwise distance uniformity
    distances = []
    for i in range(k):
        for j in range(i+1, k):
            dist = torch.norm(prototypes[i] - prototypes[j]).item()
            distances.append(dist)
    
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    dist_cv = dist_std / (dist_mean + 1e-8)  # Coefficient of variation
    
    # Check zero-mean condition
    mean_norm = torch.norm(prototypes.mean(dim=0)).item()
    
    metrics = {
        'num_classes': k,
        'norm_mean': norm_mean,
        'norm_std': norm_std,
        'norm_cv': norm_ratio,
        'similarity_mean': sim_mean,
        'similarity_std': sim_std,
        'theoretical_similarity': theoretical_sim,
        'etf_deviation': etf_deviation,
        'distance_mean': dist_mean,
        'distance_cv': dist_cv,
        'zero_mean_norm': mean_norm,
        'is_equiangular': sim_std < 0.05,  # Low std = equal angles
        'is_equal_norm': norm_ratio < 0.1,  # Low CV = equal norms
        'is_zero_mean': mean_norm < 0.1,    # Near zero mean
    }
    
    return metrics


def compute_collision_mass(features, labels, prototypes):
    """
    Compute collision mass (Definition 2) - fraction of secondary classes per cell
    """
    device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    labels = labels.to(device)
    
    distances = torch.cdist(features, prototypes)
    cell_assignments = torch.argmin(distances, dim=1)
    
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
        correct_count = (labels_in_cell == i).sum().item()
        total_in_cell = mask.sum().item()
        collision = 1 - (correct_count / total_in_cell)
        collision_per_cell[i] = collision
        total_collision += total_in_cell - correct_count
        total_samples += total_in_cell
    
    overall_collision = np.mean(list(collision_per_cell.values())) if collision_per_cell else 0
    total_collision_frac = total_collision / total_samples if total_samples > 0 else 0
    
    return overall_collision, total_collision_frac, collision_per_cell


def compute_within_class_variance(features, labels, prototypes):
    """
    Compute within-class variance (neural collapse metric)
    Lower = better collapse
    """
    device = features.device
    features = features.to(device)
    labels = labels.to(device)
    prototypes = prototypes.to(device)
    
    total_var = 0
    total_samples = 0
    
    for c in range(len(prototypes)):
        mask = (labels == c)
        if mask.sum() > 0:
            class_features = features[mask]
            # Mean squared distance to prototype
            var = torch.mean(torch.norm(class_features - prototypes[c], dim=1) ** 2)
            total_var += var.item() * mask.sum().item()
            total_samples += mask.sum().item()
    
    within_class_var = total_var / total_samples if total_samples > 0 else 0
    return within_class_var


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


# ============================================================================
# 4. SINGLE EXPERIMENT FOR ONE CONFIGURATION
# ============================================================================

def run_single_config(feature_dim, num_classes=10, epochs=100, 
                      use_neural_collapse_regime=False, 
                      use_full_dataset=False,
                      weight_decay=5e-4):
    """
    Run experiment for a single configuration.
    
    Args:
        feature_dim: dimension of feature space
        num_classes: number of classes (k)
        epochs: number of training epochs
        use_neural_collapse_regime: if True, add weight decay and normalization
        use_full_dataset: if True, use full CIFAR-10
        weight_decay: weight decay strength
    """
    
    regime_name = "neural_collapse" if use_neural_collapse_regime else "standard"
    
    print(f"\n{'='*60}")
    print(f"Testing: m={feature_dim}, k={num_classes}, regime={regime_name}")
    print(f"Theorem 7 predicts: m ≥ {num_classes - 1} for collision-free ETF")
    print(f"{'='*60}")
    
    # Data loading
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
        print(f"Full dataset: {len(train_dataset)} train, {len(test_dataset)} test")
    else:
        train_dataset = Subset(full_train, range(8000))
        test_dataset = Subset(full_test, range(2000))
        print(f"Subset: 8,000 train, 2,000 test")
    
    train_loader = DataLoader(train_dataset, batch_size=128, 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VoronoiCNN(input_channels=3, feature_dim=feature_dim, num_classes=num_classes).to(device)
    
    # Optimizer with optional weight decay for neural collapse regime
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay if use_neural_collapse_regime else 0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    best_accuracy = 0
    final_metrics = {}
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        if epoch % 20 == 0 or epoch == epochs:
            test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
            
            # Only compute collapse metrics occasionally (expensive)
            if epoch == epochs:
                prototypes, _ = compute_prototypes_from_features(features, labels, num_classes)
                collapse_metrics = compute_neural_collapse_metrics(prototypes)
                collision, total_collision, _ = compute_collision_mass(features, labels, prototypes)
                within_var = compute_within_class_variance(features, labels, prototypes)
                
                print(f"  Epoch {epoch}: Acc={test_acc:.4f}, Collision={collision:.4f}")
                print(f"    ETF deviation: {collapse_metrics['etf_deviation']:.4f}")
                print(f"    Equal norms: {collapse_metrics['is_equal_norm']}")
                print(f"    Equiangular: {collapse_metrics['is_equiangular']}")
                
                final_metrics = {
                    'test_accuracy': test_acc,
                    'collision_mass': collision,
                    'total_collision_frac': total_collision,
                    'within_class_var': within_var,
                    'collapse_metrics': collapse_metrics
                }
                
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
    
    final_metrics['best_accuracy'] = best_accuracy
    final_metrics['feature_dim'] = feature_dim
    final_metrics['num_classes'] = num_classes
    final_metrics['regime'] = regime_name
    final_metrics['epochs_trained'] = epochs
    
    print(f"\n  FINAL: m={feature_dim}, regime={regime_name}")
    print(f"    Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"    Collision: {final_metrics['collision_mass']:.4f}")
    print(f"    ETF deviation: {final_metrics['collapse_metrics']['etf_deviation']:.4f}")
    
    return final_metrics


# ============================================================================
# 5. MAIN EXPERIMENT (STEP 7)
# ============================================================================

def run_step7_experiment(dimensions=[2, 3, 4, 5, 6, 7, 8, 9, 10], 
                         epochs=100,
                         use_full_dataset=False):
    """
    Step 7: Symmetry vs Separation - Neural Collapse (Theorem 7)
    
    Tests dimensionality bound: m >= k-1 for collision-free ETF configuration.
    Compares standard training vs neural collapse regime (with weight decay).
    """
    
    save_dir = 'step7_results_neural_collapse'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Step 7: Symmetry vs Separation - Neural Collapse (Theorem 7)")
    print("="*70)
    print(f"k = 10 classes")
    print(f"Theorem 7: m ≥ k-1 = 9 for collision-free ETF configuration")
    print(f"Testing dimensions: {dimensions}")
    print(f"Epochs per dimension: {epochs}")
    print("="*70 + "\n")
    
    # Run standard training
    print("\n" + "="*50)
    print("REGIME 1: STANDARD TRAINING (no weight decay)")
    print("="*50)
    
    standard_results = []
    for dim in dimensions:
        results = run_single_config(
            feature_dim=dim,
            num_classes=10,
            epochs=epochs,
            use_neural_collapse_regime=False,
            use_full_dataset=use_full_dataset,
            weight_decay=0
        )
        standard_results.append(results)
        
        # Save intermediate
        with open(f'{save_dir}/standard_m{dim}.json', 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'collapse_metrics'}, f, indent=2)
    
    # Run neural collapse regime (with weight decay)
    print("\n" + "="*50)
    print("REGIME 2: NEURAL COLLAPSE REGIME (with weight decay + normalization)")
    print("="*50)
    
    collapse_results = []
    for dim in dimensions:
        results = run_single_config(
            feature_dim=dim,
            num_classes=10,
            epochs=epochs,
            use_neural_collapse_regime=True,
            use_full_dataset=use_full_dataset,
            weight_decay=5e-4
        )
        collapse_results.append(results)
        
        with open(f'{save_dir}/collapse_m{dim}.json', 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'collapse_metrics'}, f, indent=2)
    
    # Aggregate results
    dims = dimensions
    std_acc = [r['test_accuracy'] for r in standard_results]
    std_coll = [r['collision_mass'] for r in standard_results]
    std_etf = [r['collapse_metrics']['etf_deviation'] for r in standard_results]
    
    coll_acc = [r['test_accuracy'] for r in collapse_results]
    coll_coll = [r['collision_mass'] for r in collapse_results]
    coll_etf = [r['collapse_metrics']['etf_deviation'] for r in collapse_results]
    
    # Find critical dimensions
    std_critical = None
    coll_critical = None
    
    for i, dim in enumerate(dims):
        if std_coll[i] < 0.1 and std_critical is None:
            std_critical = dim
        if coll_coll[i] < 0.1 and coll_critical is None:
            coll_critical = dim
    
    print("\n" + "="*70)
    print("Step 7 Results Summary")
    print("="*70)
    print(f"\nTheorem 7 predicts: m ≥ 9 for collision-free ETF configuration")
    print(f"\n{'m':<6} {'Standard Acc':<14} {'Standard Coll':<14} {'Collapse Acc':<14} {'Collapse Coll':<14}")
    print("-" * 70)
    for i, dim in enumerate(dims):
        print(f"{dim:<6} {std_acc[i]:<14.4f} {std_coll[i]:<14.4f} {coll_acc[i]:<14.4f} {coll_coll[i]:<14.4f}")
    print("-" * 70)
    
    print(f"\nStandard training - Critical dimension (collision < 0.1): {std_critical if std_critical else '>10'}")
    print(f"Neural collapse regime - Critical dimension: {coll_critical if coll_critical else '>10'}")
    
    # Generate plots
    plot_step7_results(dims, std_acc, std_coll, coll_acc, coll_coll, std_etf, coll_etf,
                      std_critical, coll_critical, save_dir)
    
    # Save all results
    final_results = {
        'dimensions': dims,
        'standard': {
            'accuracy': std_acc,
            'collision': std_coll,
            'etf_deviation': std_etf,
            'critical_dim': std_critical
        },
        'neural_collapse': {
            'accuracy': coll_acc,
            'collision': coll_coll,
            'etf_deviation': coll_etf,
            'critical_dim': coll_critical
        },
        'theorem7_prediction': 9
    }
    
    torch.save(final_results, f'{save_dir}/all_results.pt')
    
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump({k: v for k, v in final_results.items() if k != 'per_dim_results'}, f, indent=2)
    
    print(f"\nResults saved to {save_dir}/")
    print("Step 7 Complete!")
    
    return final_results


# ============================================================================
# 6. PLOTTING FUNCTIONS
# ============================================================================

def plot_step7_results(dims, std_acc, std_coll, coll_acc, coll_coll, 
                       std_etf, coll_etf, std_critical, coll_critical, save_dir):
    """Create comprehensive figures for Step 7"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Collision Mass vs Dimension (KEY FIGURE)
    axes[0, 0].plot(dims, std_coll, 'bo-', linewidth=2, markersize=8, label='Standard Training')
    axes[0, 0].plot(dims, coll_coll, 'rs-', linewidth=2, markersize=8, label='Neural Collapse Regime')
    axes[0, 0].axhline(y=0.1, color='gray', linestyle='--', linewidth=2, label='Collision threshold (10%)')
    axes[0, 0].axvline(x=9, color='green', linestyle='--', linewidth=2, label='Theorem 7: m = k-1 = 9')
    if std_critical:
        axes[0, 0].axvline(x=std_critical, color='blue', linestyle=':', linewidth=2, label=f'Standard critical = {std_critical}')
    if coll_critical:
        axes[0, 0].axvline(x=coll_critical, color='red', linestyle=':', linewidth=2, label=f'Collapse critical = {coll_critical}')
    axes[0, 0].set_xlabel('Feature Dimension (m)')
    axes[0, 0].set_ylabel('Collision Mass (C)')
    axes[0, 0].set_title('Step 7.1-7.2: Collision Mass vs Dimension')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Plot 2: Test Accuracy vs Dimension
    axes[0, 1].plot(dims, std_acc, 'bo-', linewidth=2, markersize=8, label='Standard Training')
    axes[0, 1].plot(dims, coll_acc, 'rs-', linewidth=2, markersize=8, label='Neural Collapse Regime')
    axes[0, 1].axvline(x=9, color='green', linestyle='--', linewidth=2, label='Theorem 7: m = k-1 = 9')
    axes[0, 1].set_xlabel('Feature Dimension (m)')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Classification Accuracy vs Dimension')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # Plot 3: ETF Deviation vs Dimension (how close to simplex)
    axes[1, 0].plot(dims, std_etf, 'bo-', linewidth=2, markersize=8, label='Standard Training')
    axes[1, 0].plot(dims, coll_etf, 'rs-', linewidth=2, markersize=8, label='Neural Collapse Regime')
    axes[1, 0].axvline(x=9, color='green', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Feature Dimension (m)')
    axes[1, 0].set_ylabel('ETF Deviation (|sim - (-1/(k-1))|)')
    axes[1, 0].set_title('Step 7.3: Deviation from Simplex ETF')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Collision vs Accuracy correlation
    axes[1, 1].scatter(std_coll, std_acc, s=100, alpha=0.7, c='blue', label='Standard', marker='o')
    axes[1, 1].scatter(coll_coll, coll_acc, s=100, alpha=0.7, c='red', label='Neural Collapse', marker='s')
    axes[1, 1].set_xlabel('Collision Mass')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Collision-Accuracy Trade-off')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/step7_summary.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/step7_summary.png")
    
    # Also create a bar chart comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collision comparison bar chart
    x = np.arange(len(dims))
    width = 0.35
    axes2[0].bar(x - width/2, std_coll, width, label='Standard', color='blue', alpha=0.7)
    axes2[0].bar(x + width/2, coll_coll, width, label='Neural Collapse', color='red', alpha=0.7)
    axes2[0].axhline(y=0.1, color='black', linestyle='--', label='10% threshold')
    axes2[0].set_xlabel('Feature Dimension (m)')
    axes2[0].set_ylabel('Collision Mass')
    axes2[0].set_title('Collision Mass: Standard vs Neural Collapse')
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(dims)
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy comparison bar chart
    axes2[1].bar(x - width/2, std_acc, width, label='Standard', color='blue', alpha=0.7)
    axes2[1].bar(x + width/2, coll_acc, width, label='Neural Collapse', color='red', alpha=0.7)
    axes2[1].set_xlabel('Feature Dimension (m)')
    axes2[1].set_ylabel('Test Accuracy')
    axes2[1].set_title('Accuracy: Standard vs Neural Collapse')
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(dims)
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_barchart.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/comparison_barchart.png")


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 7: Neural Collapse (Theorem 7)')
    parser.add_argument('--dims', type=str, default='2,3,4,5,6,7,8,9,10',
                        help='Comma-separated list of dimensions to test')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs per dimension')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer dimensions for testing')
    
    args = parser.parse_args()
    
    if args.quick:
        dims = [2, 4, 6, 8, 10]
        epochs = 60
        print("Quick mode enabled")
    else:
        dims = [int(d) for d in args.dims.split(',')]
        epochs = args.epochs
    
    results = run_step7_experiment(
        dimensions=dims,
        epochs=epochs,
        use_full_dataset=args.full
    )
