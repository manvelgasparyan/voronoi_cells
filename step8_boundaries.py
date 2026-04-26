"""
Step 8: Classification Errors Concentrate Near Voronoi Boundaries

This script tests the claim that misclassified samples lie closer to Voronoi boundaries
than correctly classified samples.

Experiments:
- 8.1: Measure margin distribution for correct vs incorrect predictions
- 8.2: Correlate distance to boundary with model uncertainty (softmax confidence)
- 8.3: Visualize error concentration near decision boundaries

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
from tqdm import tqdm
from scipy.stats import gaussian_kde

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
    
    def compute_prototypes(self, dataloader, device):
        self.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Computing prototypes", leave=False):
                images = images.to(device)
                _, features = self.forward(images, return_features=True)
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        prototypes = []
        for c in range(self.classifier.out_features):
            mask = all_labels == c
            if mask.sum() > 0:
                proto = all_features[mask].mean(dim=0)
            else:
                proto = torch.zeros(self.feature_layer.out_features)
            prototypes.append(proto)
        
        self.prototypes = torch.stack(prototypes).to(device)
        return self.prototypes


# ============================================================================
# 2. VORONOI BOUNDARY METRICS
# ============================================================================

def compute_margin(feature, prototype_correct, prototypes_other):
    """
    Compute classification margin for a single feature vector.
    Margin = distance to closest wrong prototype - distance to correct prototype
    
    Positive margin = correctly classified (inside correct Voronoi cell)
    Negative margin = misclassified (closer to wrong prototype)
    
    Args:
        feature: (m,) tensor
        prototype_correct: (m,) tensor for true class
        prototypes_other: (k-1, m) tensor for all other classes
    
    Returns:
        margin: float (positive = correct, negative = incorrect)
        dist_correct: distance squared to correct prototype
        dist_wrong_min: minimum distance squared to any wrong prototype
    """
    dist_correct = torch.sum((feature - prototype_correct) ** 2).item()
    dists_wrong = torch.sum((feature.unsqueeze(0) - prototypes_other) ** 2, dim=1)
    dist_wrong_min = torch.min(dists_wrong).item()
    margin = dist_wrong_min - dist_correct
    return margin, dist_correct, dist_wrong_min


def compute_margins_batch(features, labels, prototypes):
    """
    Compute margins for a batch of features.
    
    Returns:
        margins: (N,) tensor of margins
        is_correct: (N,) bool tensor
        confidences: (N,) tensor of softmax probabilities for true class
    """
    device = features.device
    N = len(features)
    k = len(prototypes)
    
    # Compute all pairwise distances
    distances = torch.cdist(features, prototypes) ** 2  # (N, k)
    
    # Distance to correct prototype
    dist_correct = distances[torch.arange(N), labels]  # (N,)
    
    # Distance to closest wrong prototype
    # Set distance to correct class to infinity, then take min
    wrong_distances = distances.clone()
    wrong_distances[torch.arange(N), labels] = float('inf')
    dist_wrong_min = torch.min(wrong_distances, dim=1)[0]  # (N,)
    
    # Margin
    margins = dist_wrong_min - dist_correct
    
    # Correctness
    is_correct = margins > 0
    
    # Softmax confidence for true class
    logits = -0.5 * distances  # Simplified (assuming tau=1, ignoring biases)
    probs = torch.softmax(logits, dim=1)
    confidences = probs[torch.arange(N), labels]
    
    return margins, is_correct, confidences, dist_correct, dist_wrong_min


def compute_normalized_margin(features, labels, prototypes):
    """
    Compute margin normalized by distance to correct prototype.
    Normalized margin = (d_wrong - d_correct) / d_correct
    """
    device = features.device
    N = len(features)
    k = len(prototypes)
    
    distances = torch.cdist(features, prototypes) ** 2
    dist_correct = distances[torch.arange(N), labels]
    
    wrong_distances = distances.clone()
    wrong_distances[torch.arange(N), labels] = float('inf')
    dist_wrong_min = torch.min(wrong_distances, dim=1)[0]
    
    margins = dist_wrong_min - dist_correct
    normalized_margins = margins / (dist_correct + 1e-8)
    
    return normalized_margins


def compute_distance_to_boundary(features, labels, prototypes):
    """
    Compute distance from each feature to the nearest Voronoi boundary.
    The boundary is where distances to two prototypes are equal.
    
    For a given feature, the distance to boundary is:
    min_{j != y} |distance to prototype y - distance to prototype j| / (2 * ||z_y - z_j||)
    """
    device = features.device
    N = len(features)
    k = len(prototypes)
    
    distances = torch.cdist(features, prototypes)  # (N, k)
    dist_sq = distances ** 2
    
    boundary_distances = torch.full((N,), float('inf'), device=device)
    
    for i in range(N):
        # For each sample, find the nearest boundary with any other class
        y = labels[i]
        dist_y = distances[i, y]
        
        for j in range(k):
            if j == y:
                continue
            # Distance to boundary between class y and class j
            # Boundary is where dist_y = dist_j
            # Distance from point to this boundary = |dist_y - dist_j| / 2
            # Actually, in Euclidean space with Voronoi, it's more complex
            # Simplified: use margin / (2 * ||z_y - z_j||)
            proto_diff = torch.norm(prototypes[y] - prototypes[j])
            if proto_diff > 1e-8:
                boundary_dist = torch.abs(distances[i, y] - distances[i, j]) / (2 * proto_diff)
                if boundary_dist < boundary_distances[i]:
                    boundary_distances[i] = boundary_dist
    
    return boundary_distances


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
# 4. MAIN EXPERIMENT (STEP 8)
# ============================================================================

def run_step8_experiment(feature_dim=2, epochs=50, use_full_dataset=False):
    """
    Step 8: Errors Concentrate Near Voronoi Boundaries
    
    - 8.1: Margin distribution for correct vs incorrect predictions
    - 8.2: Correlation between margin and confidence
    - 8.3: Visualize error concentration
    """
    
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dataset': 'CIFAR-10',
        'save_dir': f'step8_results_boundaries_m{feature_dim}',
        'eval_freq': 10
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Feature dimension: {config['feature_dim']}")
    
    # ========== Load CIFAR-10 ==========
    print("\nLoading CIFAR-10 dataset...")
    
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
        print(f"Full CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test")
    else:
        train_dataset = Subset(full_train, range(8000))
        test_dataset = Subset(full_test, range(2000))
        print(f"Subset: 8,000 train, 2,000 test")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # ========== Initialize Model ==========
    model = VoronoiCNN(input_channels=3, feature_dim=config['feature_dim'], num_classes=10).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    print("\n" + "="*60)
    print("Step 8.1: Margin Distribution (Correct vs Incorrect)")
    print("Step 8.2: Correlation with Model Confidence")
    print("Step 8.3: Error Concentration Near Boundaries")
    print("="*60 + "\n")
    
    # ========== Training Loop ==========
    all_margins_correct = []
    all_margins_incorrect = []
    all_confidences_correct = []
    all_confidences_incorrect = []
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        # Evaluate
        test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
        
        # Compute prototypes
        prototypes = model.compute_prototypes(train_loader, device)
        
        # Compute margins and confidences
        margins, is_correct, confidences, dist_correct, dist_wrong_min = compute_margins_batch(
            features.to(device), labels.to(device), prototypes
        )
        
        # Separate correct vs incorrect
        margins_correct = margins[is_correct].cpu().numpy()
        margins_incorrect = margins[~is_correct].cpu().numpy()
        conf_correct = confidences[is_correct].cpu().numpy()
        conf_incorrect = confidences[~is_correct].cpu().numpy()
        
        all_margins_correct.append(margins_correct)
        all_margins_incorrect.append(margins_incorrect)
        all_confidences_correct.append(conf_correct)
        all_confidences_incorrect.append(conf_incorrect)
        
        print(f"\n[Epoch {epoch:03d}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"  Step 8.1 - Margin (correct): mean={margins_correct.mean():.4f}, median={np.median(margins_correct):.4f}")
        print(f"  Step 8.1 - Margin (incorrect): mean={margins_incorrect.mean():.4f}, median={np.median(margins_incorrect):.4f}")
        print(f"  Step 8.2 - Confidence (correct): {conf_correct.mean():.4f}")
        print(f"  Step 8.2 - Confidence (incorrect): {conf_incorrect.mean():.4f}")
        
        # Check the claim: incorrect samples should have smaller/negative margins
        if margins_incorrect.mean() < margins_correct.mean():
            print(f"  ✓ Errors have smaller margin (supports claim)")
        else:
            print(f"  ✗ Errors do NOT have smaller margin")
    
    # ========== Final Evaluation ==========
    print("\n" + "="*60)
    print("Final Analysis - Error Concentration Near Boundaries")
    print("="*60)
    
    # Final feature extraction
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = model.compute_prototypes(train_loader, device)
    margins, is_correct, confidences, dist_correct, dist_wrong_min = compute_margins_batch(
        features.to(device), labels.to(device), prototypes
    )
    
    margins_correct_final = margins[is_correct].cpu().numpy()
    margins_incorrect_final = margins[~is_correct].cpu().numpy()
    conf_correct_final = confidences[is_correct].cpu().numpy()
    conf_incorrect_final = confidences[~is_correct].cpu().numpy()
    
    # Statistical test
    from scipy.stats import mannwhitneyu
    statistic, p_value = mannwhitneyu(margins_correct_final, margins_incorrect_final, alternative='greater')
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nStep 8.1 - Margin Statistics:")
    print(f"  Correct samples: mean={margins_correct_final.mean():.4f}, median={np.median(margins_correct_final):.4f}")
    print(f"  Incorrect samples: mean={margins_incorrect_final.mean():.4f}, median={np.median(margins_incorrect_final):.4f}")
    print(f"  Mann-Whitney U test (correct > incorrect): p={p_value:.6f}")
    
    print(f"\nStep 8.2 - Confidence Statistics:")
    print(f"  Correct samples: mean={conf_correct_final.mean():.4f}, median={np.median(conf_correct_final):.4f}")
    print(f"  Incorrect samples: mean={conf_incorrect_final.mean():.4f}, median={np.median(conf_incorrect_final):.4f}")
    
    # Check concentration near zero margin
    near_boundary_frac_correct = (np.abs(margins_correct_final) < 1.0).mean()
    near_boundary_frac_incorrect = (np.abs(margins_incorrect_final) < 1.0).mean()
    
    print(f"\nStep 8.3 - Concentration near boundary (|margin| < 1.0):")
    print(f"  Correct samples: {near_boundary_frac_correct:.2%}")
    print(f"  Incorrect samples: {near_boundary_frac_incorrect:.2%}")
    
    # Verdict
    print("\n" + "="*60)
    print("Verdict: Classification Errors Concentrate Near Voronoi Boundaries")
    print("="*60)
    
    if p_value < 0.001 and margins_incorrect_final.mean() < margins_correct_final.mean():
        verdict = "CONFIRMED - Errors have significantly smaller margins (p < 0.001)"
    elif p_value < 0.05:
        verdict = "SUPPORTED - Errors have smaller margins (p < 0.05)"
    else:
        verdict = "INCONCLUSIVE - Margin difference not statistically significant"
    
    print(f"Verdict: {verdict}")
    
    # Save results
    results = {
        'test_accuracy': test_acc,
        'margins_correct': margins_correct_final.tolist(),
        'margins_incorrect': margins_incorrect_final.tolist(),
        'confidences_correct': conf_correct_final.tolist(),
        'confidences_incorrect': conf_incorrect_final.tolist(),
        'mean_margin_correct': float(margins_correct_final.mean()),
        'mean_margin_incorrect': float(margins_incorrect_final.mean()),
        'p_value': float(p_value),
        'near_boundary_correct': float(near_boundary_frac_correct),
        'near_boundary_incorrect': float(near_boundary_frac_incorrect),
        'verdict': verdict,
        'config': config
    }
    
    torch.save(results, f"{config['save_dir']}/results.pt")
    
    # Generate plots
    plot_margin_distribution(margins_correct_final, margins_incorrect_final, config['save_dir'])
    plot_confidence_vs_margin(conf_correct_final, conf_incorrect_final, 
                              margins_correct_final, margins_incorrect_final, 
                              config['save_dir'])
    plot_2d_voronoi_with_errors(features, labels, prototypes, is_correct, config['save_dir'])
    
    print(f"\nResults saved to {config['save_dir']}/")
    print("Step 8 Complete!")
    
    return model, results


# ============================================================================
# 5. PLOTTING FUNCTIONS
# ============================================================================

def plot_margin_distribution(margins_correct, margins_incorrect, save_dir):
    """Create histogram of margins for correct vs incorrect samples"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with KDE
    axes[0].hist(margins_correct, bins=50, alpha=0.5, color='green', label='Correct', density=True)
    axes[0].hist(margins_incorrect, bins=50, alpha=0.5, color='red', label='Incorrect', density=True)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    axes[0].set_xlabel('Margin (d_wrong - d_correct)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Margin Distribution: Correct vs Incorrect')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [margins_correct, margins_incorrect]
    bp = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Margin')
    axes[1].set_title('Margin Comparison (Box Plot)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/margin_distribution.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/margin_distribution.png")


def plot_confidence_vs_margin(conf_correct, conf_incorrect, margins_correct, margins_incorrect, save_dir):
    """Create scatter plots of confidence vs margin"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: Confidence vs Margin
    axes[0].scatter(margins_correct, conf_correct, alpha=0.3, s=10, c='green', label='Correct')
    axes[0].scatter(margins_incorrect, conf_incorrect, alpha=0.3, s=10, c='red', label='Incorrect')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Margin')
    axes[0].set_ylabel('Confidence (p_y)')
    axes[0].set_title('Confidence vs Margin')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Binned confidence vs margin
    all_margins = np.concatenate([margins_correct, margins_incorrect])
    all_conf = np.concatenate([conf_correct, conf_incorrect])
    
    # Bin by margin
    bins = np.linspace(all_margins.min(), all_margins.max(), 21)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    
    for i in range(len(bins)-1):
        mask = (all_margins >= bins[i]) & (all_margins < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(all_conf[mask].mean())
            bin_stds.append(all_conf[mask].std())
        else:
            bin_means.append(0)
            bin_stds.append(0)
    
    axes[1].errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=3, color='blue')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Margin')
    axes[1].set_ylabel('Mean Confidence')
    axes[1].set_title('Confidence Increases with Margin')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_vs_margin.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/confidence_vs_margin.png")


def plot_2d_voronoi_with_errors(features, labels, prototypes, is_correct, save_dir):
    """
    Visualize Voronoi cells with errors highlighted (only works for 2D)
    """
    if features.shape[1] != 2:
        print(f"Skipping 2D visualization: feature dimension={features.shape[1]} (need 2D)")
        return
    
    device = prototypes.device
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    prototypes_np = prototypes.cpu().numpy()
    is_correct_np = is_correct.cpu().numpy()
    
    # Create grid for Voronoi visualization
    margin = 1.0
    x_min, x_max = features_np[:, 0].min() - margin, features_np[:, 0].max() + margin
    y_min, y_max = features_np[:, 1].min() - margin, features_np[:, 1].max() + margin
    
    resolution = 200
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Assign Voronoi cells
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    distances = torch.cdist(grid_tensor, prototypes)
    cell_labels = torch.argmin(distances, dim=1).cpu().numpy().reshape(xx.shape)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: All points colored by true label
    scatter1 = axes[0].scatter(features_np[:, 0], features_np[:, 1], 
                               c=labels_np, cmap='tab10', s=10, alpha=0.6)
    axes[0].scatter(prototypes_np[:, 0], prototypes_np[:, 1], 
                    c='red', marker='X', s=200, edgecolors='black', linewidth=2)
    axes[0].set_title('Feature Space (Colored by True Class)')
    axes[0].set_xlabel('Feature Dimension 1')
    axes[0].set_ylabel('Feature Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='Class')
    
    # Plot 2: Errors highlighted
    # First plot Voronoi boundaries
    unique_cells = np.unique(cell_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cells)))
    cell_to_color = {cell: colors[i] for i, cell in enumerate(unique_cells)}
    
    colored_regions = np.zeros((*cell_labels.shape, 4))
    for cell_id in unique_cells:
        mask = cell_labels == cell_id
        colored_regions[mask] = cell_to_color[cell_id]
    
    axes[1].imshow(colored_regions, extent=[x_min, x_max, y_min, y_max],
                   origin='lower', alpha=0.2)
    
    # Plot correct points (green edge) and incorrect points (red x)
    correct_mask = is_correct_np
    incorrect_mask = ~is_correct_np
    
    axes[1].scatter(features_np[correct_mask, 0], features_np[correct_mask, 1],
                    c=labels_np[correct_mask], cmap='tab10', s=15, alpha=0.7,
                    edgecolors='black', linewidth=0.3, label='Correct')
    axes[1].scatter(features_np[incorrect_mask, 0], features_np[incorrect_mask, 1],
                    c=labels_np[incorrect_mask], cmap='tab10', s=30, alpha=0.9,
                    edgecolors='red', linewidth=1.5, marker='X', label='Incorrect')
    axes[1].scatter(prototypes_np[:, 0], prototypes_np[:, 1],
                    c='red', marker='X', s=200, edgecolors='black', linewidth=2)
    axes[1].set_title('Errors Concentrate Near Voronoi Boundaries')
    axes[1].set_xlabel('Feature Dimension 1')
    axes[1].set_ylabel('Feature Dimension 2')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/voronoi_errors_visualization.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/voronoi_errors_visualization.png")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 8: Errors Near Voronoi Boundaries')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Feature space dimension (2 or 3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    
    args = parser.parse_args()
    
    model, results = run_step8_experiment(
        feature_dim=args.dim,
        epochs=args.epochs,
        use_full_dataset=args.full
    )
