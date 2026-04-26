"""
Step 1: Verifying Voronoi Structure in Learned Representations (CIFAR-10)

This script trains a CNN on CIFAR-10 with low-dimensional feature space (2D or 3D)
and visualizes the resulting Voronoi tessellation, measuring cell purity over time.

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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. MODEL ARCHITECTURE FOR CIFAR-10
# ============================================================================

class VoronoiCNN(nn.Module):
    """
    CNN for CIFAR-10 with configurable feature dimension m.
    Outputs: features (m-dim) and logits (k-dim)
    """
    def __init__(self, input_channels=3, feature_dim=2, num_classes=10):
        super(VoronoiCNN, self).__init__()
        
        # Feature extractor for CIFAR-10 (32x32 input)
        self.feature_extractor = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Flatten + feature projection to m-dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Feature layer (output dimension m)
        self.feature_layer = nn.Linear(256, feature_dim)
        
        # Register buffers for prototypes
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        
        # Classifier (for training only)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # Extract features
        feats = self.feature_extractor(x)
        feats = self.flatten(feats)
        feats = self.fc1(feats)
        feats = self.bn1(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        feats = self.fc2(feats)
        feats = self.bn2(feats)
        feats = self.relu(feats)
        features = self.feature_layer(feats)  # shape: (batch, m)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def compute_prototypes(self, dataloader, device):
        """Compute class prototypes as mean features per class"""
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
# 2. VORONOI UTILITIES
# ============================================================================
 

def assign_voronoi_cells(features, prototypes, device=None):
    """
    Assign each feature vector to its nearest prototype (Voronoi cell)
    
    Args:
        features: (N, m) tensor of feature vectors (can be CPU or GPU)
        prototypes: (k, m) tensor of class prototypes (can be CPU or GPU)
        device: device to run computation on (optional)
    
    Returns:
        cell_assignments: (N,) tensor of cell indices (0..k-1)
    """
    # Move both to same device
    if device is None:
        device = prototypes.device
    
    features = features.to(device)
    prototypes = prototypes.to(device)
    
    distances = torch.cdist(features, prototypes)
    cell_assignments = torch.argmin(distances, dim=1)
    return cell_assignments


def compute_cell_purity(features, labels, prototypes):
    """
    For each Voronoi cell, compute what fraction of its samples belong to
    the majority class (label consistency).
    
    Returns:
        purity_per_cell: dict cell_id -> purity (0 to 1)
        label_distribution: dict cell_id -> dict label -> count
    """
    device = prototypes.device
    cell_assignments = assign_voronoi_cells(features, prototypes, device=device)
    
    # Move to CPU for numpy operations
    cell_assignments = cell_assignments.cpu()
    features = features.cpu()
    labels = labels.cpu()
    
    label_distribution = defaultdict(lambda: defaultdict(int))
    
    for cell_idx, label in zip(cell_assignments.tolist(), labels.tolist()):
        label_distribution[cell_idx][label] += 1
    
    purity_per_cell = {}
    for cell_idx, dist in label_distribution.items():
        total = sum(dist.values())
        majority_count = max(dist.values())
        purity_per_cell[cell_idx] = majority_count / total if total > 0 else 0
    
    return purity_per_cell, label_distribution


def compute_overall_cell_purity(features, labels, prototypes):
    """Compute macro-averaged purity across all occupied cells"""
    purity_per_cell, _ = compute_cell_purity(features, labels, prototypes)
    if len(purity_per_cell) > 0:
        return np.mean(list(purity_per_cell.values()))
    return 0.0


def compute_cell_collision_mass(features, labels, prototypes):
    """
    Compute collision mass per cell (Definition 2 in paper):
    Total measure of secondary classes within a cell designated for primary class
    
    Returns:
        collision_mass: dict cell_id -> collision mass
        overall_collision: float
    """
    device = prototypes.device
    cell_assignments = assign_voronoi_cells(features, prototypes, device=device)
    
    # Move to CPU for numpy operations
    cell_assignments = cell_assignments.cpu()
    labels = labels.cpu()
    
    collision_per_cell = {}
    for cell_idx in range(len(prototypes)):
        mask = cell_assignments == cell_idx
        if mask.sum() == 0:
            collision_per_cell[cell_idx] = 0.0
            continue
        
        labels_in_cell = labels[mask]
        # Primary class for this cell is cell_idx (by prototype assignment)
        secondary_count = (labels_in_cell != cell_idx).sum().item()
        collision_per_cell[cell_idx] = secondary_count / mask.sum().item()
    
    overall_collision = np.mean(list(collision_per_cell.values()))
    return collision_per_cell, overall_collision


def get_voronoi_boundary_points(prototypes, bounds=(-3, 3), resolution=200):
    """
    For 2D feature space: generate grid points and color by Voronoi cell.
    Returns grid and cell labels for visualization.
    """
    if prototypes.shape[1] != 2:
        raise ValueError("Voronoi boundary visualization requires 2D feature space")
    
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Convert to torch and assign cells
    device = prototypes.device
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    proto_tensor = prototypes.to(device)
    distances = torch.cdist(grid_tensor, proto_tensor)
    cell_labels = torch.argmin(distances, dim=1).cpu().numpy().reshape(xx.shape)
    
    return xx, yy, cell_labels


def plot_voronoi_with_data(prototypes, features, labels, epoch, save_dir='voronoi_plots'):
    """Create Voronoi diagram with data points overlaid"""
    os.makedirs(save_dir, exist_ok=True)
    
    if prototypes.shape[1] != 2:
        print(f"Skipping visualization: feature dimension={prototypes.shape[1]} (need 2D)")
        return
    
    # Move everything to CPU for plotting
    prototypes_np = prototypes.detach().cpu().numpy()
    features_np = features.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Dynamically determine bounds based on data
    margin = 0.5
    x_min, x_max = features_np[:, 0].min() - margin, features_np[:, 0].max() + margin
    y_min, y_max = features_np[:, 1].min() - margin, features_np[:, 1].max() + margin
    
    # Compute Voronoi boundaries (pass prototypes to device-aware function)
    xx, yy, cell_labels = get_voronoi_boundary_points(
        prototypes, bounds=(min(x_min, y_min), max(x_max, y_max)), resolution=200
    )
    
    # Compute cell assignments for data points
    device = prototypes.device
    cell_assignments = assign_voronoi_cells(features.to(device), prototypes, device=device).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Voronoi cells (colored by cell ID)
    unique_cells = np.unique(cell_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cells)))
    cell_to_color = {cell: colors[i] for i, cell in enumerate(unique_cells)}
    
    # Create colored region map
    colored_regions = np.zeros((*cell_labels.shape, 4))
    for cell_id in unique_cells:
        mask = cell_labels == cell_id
        colored_regions[mask] = cell_to_color[cell_id]
    
    axes[0].imshow(colored_regions, extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                   origin='lower', alpha=0.3)
    axes[0].scatter(features_np[:, 0], features_np[:, 1], c=labels_np, 
                    cmap='tab10', s=8, alpha=0.6, edgecolors='black', linewidth=0.2)
    axes[0].scatter(prototypes_np[:, 0], prototypes_np[:, 1], 
                    c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                    label='Prototypes')
    axes[0].set_title(f'Voronoi Cells + Data (Epoch {epoch})')
    axes[0].set_xlabel('Feature Dimension 1')
    axes[0].set_ylabel('Feature Dimension 2')
    axes[0].legend(loc='upper right')
    
    # Plot 2: Data points colored by true label
    scatter2 = axes[1].scatter(features_np[:, 0], features_np[:, 1], 
                               c=labels_np, cmap='tab10', s=8, alpha=0.6)
    axes[1].scatter(prototypes_np[:, 0], prototypes_np[:, 1], 
                    c='red', marker='X', s=200, edgecolors='black', linewidth=2)
    axes[1].set_title(f'Feature Space by True Label (Epoch {epoch})')
    axes[1].set_xlabel('Feature Dimension 1')
    axes[1].set_ylabel('Feature Dimension 2')
    plt.colorbar(scatter2, ax=axes[1], label='Class Label')
    
    # Plot 3: Data points colored by Voronoi cell assignment
    scatter3 = axes[2].scatter(features_np[:, 0], features_np[:, 1], 
                               c=cell_assignments, cmap='tab10', s=8, alpha=0.6)
    axes[2].scatter(prototypes_np[:, 0], prototypes_np[:, 1], 
                    c='red', marker='X', s=200, edgecolors='black', linewidth=2)
    axes[2].set_title(f'Voronoi Cell Assignment (Epoch {epoch})')
    axes[2].set_xlabel('Feature Dimension 1')
    axes[2].set_ylabel('Feature Dimension 2')
    plt.colorbar(scatter3, ax=axes[2], label='Voronoi Cell ID')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/voronoi_epoch_{epoch:04d}.png', dpi=150)
    plt.close()

# ============================================================================
# 3. TRAINING LOOP WITH METRIC TRACKING
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
# 4. MAIN EXPERIMENT (Step 1.1 and 1.2 for CIFAR-10)
# ============================================================================

def run_step1_experiment(feature_dim=2, epochs=50, use_full_dataset=False):
    """
    Main function to run Step 1 experiments
    
    Args:
        feature_dim: dimension of feature space (2 or 3 recommended)
        epochs: number of training epochs
        use_full_dataset: if True, use full CIFAR-10 (50k train, 10k test)
    """
    
    # Configuration
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dataset': 'CIFAR-10',
        'save_dir': f'step1_results_cifar10_m{feature_dim}',
        'eval_freq': 5
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Feature dimension: {config['feature_dim']}")
    print(f"Dataset: CIFAR-10")
    
    # ========== Load CIFAR-10 Dataset ==========
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
        print(f"Using full CIFAR-10: {len(train_dataset)} train, {len(test_dataset)} test")
    else:
        # Use subset for faster experimentation (10k train, 2k test)
        train_dataset = torch.utils.data.Subset(full_train, range(10000))
        test_dataset = torch.utils.data.Subset(full_test, range(2000))
        print(f"Using subset: 10,000 train, 2,000 test")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # ========== Initialize Model ==========
    model = VoronoiCNN(
        input_channels=3,
        feature_dim=config['feature_dim'],
        num_classes=config['num_classes']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # ========== Tracking Variables ==========
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    cell_purities = []
    collision_masses = []
    epoch_list = []
    
    print("\n" + "="*60)
    print("Step 1.1: Visualization of Voronoi Cells")
    print("Step 1.2: Quantifying Label Purity of Voronoi Cells")
    print("="*60 + "\n")
    
    # ========== Training Loop ==========
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        scheduler.step()
        
        # Test
        test_acc = evaluate(model, test_loader, device, compute_features=False)
        test_accuracies.append(test_acc)
        
        # Periodic evaluation for Voronoi metrics
        if epoch % config['eval_freq'] == 0 or epoch == 1 or epoch == config['epochs']:
            # Get features and labels from test set
            _, features, labels = evaluate(model, test_loader, device, compute_features=True)
            
            # Compute prototypes
            prototypes = model.compute_prototypes(train_loader, device)
            
            # Step 1.1: Visualize Voronoi cells
            if feature_dim == 2:
                plot_voronoi_with_data(prototypes, features, labels, epoch, 
                                       save_dir=config['save_dir'])
            elif feature_dim == 3:
                plot_3d_voronoi(prototypes, features, labels, epoch,
                                save_dir=config['save_dir'])
            
            # Step 1.2: Compute cell purity
            purity = compute_overall_cell_purity(features, labels, prototypes)
            cell_purities.append(purity)
            
            # Extra: Compute collision mass (Definition 2)
            _, collision = compute_cell_collision_mass(features, labels, prototypes)
            collision_masses.append(collision)
            epoch_list.append(epoch)
            
            # Per-cell purity for detailed logging
            purity_per_cell, _ = compute_cell_purity(features, labels, prototypes)
            
            print(f"\n[Epoch {epoch:03d}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f} | "
                  f"Purity: {purity:.4f} | "
                  f"Collision: {collision:.4f}")
            
            print(f"  Per-cell purity: {[round(purity_per_cell.get(i, 0), 3) for i in range(10)]}")
        
        else:
            # Light logging
            if epoch % 10 == 0:
                print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # ========== Final Evaluation and Plotting ==========
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    
    # Final feature extraction
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = model.compute_prototypes(train_loader, device)
    final_purity = compute_overall_cell_purity(features, labels, prototypes)
    _, final_collision = compute_cell_collision_mass(features, labels, prototypes)
    
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Cell Purity: {final_purity:.4f}")
    print(f"Final Collision Mass: {final_collision:.4f}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'cell_purities': cell_purities,
        'collision_masses': collision_masses,
        'epoch_list': epoch_list,
        'final_purity': final_purity,
        'final_collision': final_collision,
        'final_accuracy': test_acc,
        'config': config
    }
    
    torch.save(results, f"{config['save_dir']}/results.pt")
    
    # Generate summary plots
    plot_summary_results(train_losses, train_accuracies, test_accuracies, 
                         cell_purities, collision_masses, epoch_list, config['save_dir'])
    
    print(f"\nResults saved to {config['save_dir']}/")
    print("Done!")
    
    return model, results


def plot_summary_results(train_losses, train_accuracies, test_accuracies, 
                         cell_purities, collision_masses, purity_epochs, save_dir):
    """Create summary figure showing training curves, purity, and collision mass evolution"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over time
    axes[0, 1].plot(train_accuracies, 'g-', label='Train', linewidth=2)
    axes[0, 1].plot(test_accuracies, 'r-', label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cell Purity over time (Step 1.2 key result)
    axes[0, 2].plot(purity_epochs, cell_purities, 'purple', marker='o', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Cell Purity')
    axes[0, 2].set_title('Step 1.2: Voronoi Cell Label Purity')
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Collision Mass over time
    axes[1, 0].plot(purity_epochs, collision_masses, 'orange', marker='s', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Collision Mass')
    axes[1, 0].set_title('Definition 2: Collision Mass (Secondary Classes per Cell)')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='Perfect (0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Accuracy vs Purity correlation
    # Interpolate purity to match accuracy indices
    acc_at_purity_epochs = [test_accuracies[e-1] for e in purity_epochs]
    axes[1, 1].scatter(acc_at_purity_epochs, cell_purities, s=100, alpha=0.7, c='darkorange')
    axes[1, 1].set_xlabel('Test Accuracy')
    axes[1, 1].set_ylabel('Cell Purity')
    axes[1, 1].set_title('Correlation: Accuracy vs Voronoi Purity')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add trend line
    if len(cell_purities) > 1:
        z = np.polyfit(acc_at_purity_epochs, cell_purities, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(acc_at_purity_epochs), max(acc_at_purity_epochs), 100)
        axes[1, 1].plot(x_trend, p(x_trend), '--', color='gray', alpha=0.7)
    
    # Plot 6: Purity vs Collision (inverse relationship expected)
    axes[1, 2].scatter(cell_purities, collision_masses, s=100, alpha=0.7, c='teal')
    axes[1, 2].set_xlabel('Cell Purity')
    axes[1, 2].set_ylabel('Collision Mass')
    axes[1, 2].set_title('Purity vs Collision (Should be Anti-Correlated)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/summary_results.png', dpi=150)
    plt.close()


# ============================================================================
# 5. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 1: Voronoi Cells with CIFAR-10')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Feature space dimension (2 or 3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset (50k train, 10k test)')
    
    args = parser.parse_args()
    
    # Run experiment
    model, results = run_step1_experiment(
        feature_dim=args.dim,
        epochs=args.epochs,
        use_full_dataset=args.full
    )
