"""
Step 3: Tracking Voronoi Quantization Energy During Training

This script:
- 3.1: Tracks Voronoi quantization energy E = sum_i ∫_{V_i} ||z - z_i||² dν(z)
- 3.2: Correlates energy drop with accuracy gain

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
import matplotlib
matplotlib.use('Agg') 

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


def compute_voronoi_energy(features, labels, prototypes):
    """
    Compute Voronoi quantization energy:
    E = sum_i ∫_{V_i} ||z - z_i||² dν(z)
    
    Args:
        features: (N, m) tensor of feature vectors
        labels: (N,) tensor of true labels (not used, only for reference)
        prototypes: (k, m) tensor of class prototypes
    
    Returns:
        total_energy: float
        per_cell_energy: dict cell_id -> energy
    """
    device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    
    cell_assignments = assign_voronoi_cells(features, prototypes, device=device)
    
    k = len(prototypes)
    per_cell_energy = {}
    total_energy = 0.0
    
    for i in range(k):
        mask = (cell_assignments == i)
        if mask.sum() > 0:
            cell_features = features[mask]
            # Sum of squared distances to prototype
            energy = torch.sum(torch.norm(cell_features - prototypes[i], dim=1) ** 2)
            per_cell_energy[i] = energy.item()
            total_energy += energy.item()
    
    return total_energy, per_cell_energy


def compute_normalized_voronoi_energy(features, labels, prototypes):
    """
    Compute normalized Voronoi energy (divided by number of samples)
    for easier comparison across different dataset sizes.
    """
    total_energy, per_cell_energy = compute_voronoi_energy(features, labels, prototypes)
    normalized_energy = total_energy / len(features)
    return normalized_energy, per_cell_energy


def compute_per_class_compactness(features, labels, prototypes):
    """
    Compute how tightly each class clusters around its prototype.
    Lower = more compact (better).
    
    Returns:
        per_class_std: dict class_id -> std
        overall_std: float
    """
    device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    labels = labels.to(device)
    
    per_class_std = {}
    
    for c in range(len(prototypes)):
        mask = (labels == c)
        if mask.sum() > 0:
            class_features = features[mask]
            distances = torch.norm(class_features - prototypes[c], dim=1)
            per_class_std[c] = distances.std().item()
        else:
            per_class_std[c] = 0.0
    
    overall_std = np.mean(list(per_class_std.values()))
    return per_class_std, overall_std


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
# 4. MAIN EXPERIMENT (STEP 3)
# ============================================================================

def run_step3_experiment(feature_dim=2, epochs=50, use_full_dataset=False):
    """
    Step 3: Tracking Voronoi Quantization Energy
    
    - 3.1: Tracks Voronoi energy E = sum_i ∫_{V_i} ||z - z_i||² dν(z)
    - 3.2: Correlates energy drop with accuracy gain
    """
    
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dataset': 'CIFAR-10',
        'save_dir': f'step3_results_energy_m{feature_dim}',
        'eval_freq': 5
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
        train_dataset = Subset(full_train, range(10000))
        test_dataset = Subset(full_test, range(2000))
        print(f"Subset: 10,000 train, 2,000 test")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # ========== Initialize Model ==========
    model = VoronoiCNN(input_channels=3, feature_dim=config['feature_dim'], num_classes=10).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # ========== Tracking Variables ==========
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Step 3.1 tracking
    voronoi_energies = []      # Total quantization energy
    normalized_energies = []   # Energy per sample
    per_class_compactness = [] # Within-class std dev
    epoch_list = []
    
    # Step 3.2 tracking
    accuracy_gains = []
    energy_drops = []
    
    print("\n" + "="*60)
    print("Step 3.1: Tracking Voronoi Quantization Energy")
    print("Step 3.2: Correlating Energy Drop with Accuracy Gain")
    print("="*60 + "\n")
    
    prev_accuracy = None
    prev_energy = None
    
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
        
        # Periodic evaluation
        if epoch % config['eval_freq'] == 0 or epoch == 1 or epoch == config['epochs']:
            # Get features
            _, features, labels = evaluate(model, test_loader, device, compute_features=True)
            
            # Compute prototypes from training data
            prototypes = model.compute_prototypes(train_loader, device)
            
            # Step 3.1: Compute Voronoi energy
            total_energy, per_cell_energy = compute_voronoi_energy(features, labels, prototypes)
            norm_energy, _ = compute_normalized_voronoi_energy(features, labels, prototypes)
            
            # Compute per-class compactness
            per_class_std, overall_std = compute_per_class_compactness(features, labels, prototypes)
            
            voronoi_energies.append(total_energy)
            normalized_energies.append(norm_energy)
            per_class_compactness.append(overall_std)
            epoch_list.append(epoch)
            
            # Step 3.2: Compute energy drop and accuracy gain
            if prev_energy is not None:
                energy_drop = prev_energy - total_energy
                accuracy_gain = test_acc - prev_accuracy
                energy_drops.append(energy_drop)
                accuracy_gains.append(accuracy_gain)
            else:
                energy_drops.append(0)
                accuracy_gains.append(0)
            
            prev_energy = total_energy
            prev_accuracy = test_acc
            
            print(f"\n[Epoch {epoch:03d}]")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            print(f"  Step 3.1 - Voronoi Energy: {total_energy:.2f}")
            print(f"    Normalized (per sample): {norm_energy:.4f}")
            print(f"    Per-class compactness (std): {overall_std:.4f}")
            print(f"  Step 3.2 - Energy drop: {energy_drops[-1]:.2f} | Accuracy gain: {accuracy_gains[-1]:.4f}")
            
            # Print per-cell energy (top 3 highest)
            sorted_cells = sorted(per_cell_energy.items(), key=lambda x: x[1], reverse=True)
            print(f"    Highest energy cells: {sorted_cells[:3]}")
        
        else:
            if epoch % 10 == 0:
                print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # ========== Final Evaluation ==========
    print("\n" + "="*60)
    print("Final Results - Voronoi Quantization Energy")
    print("="*60)
    
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = model.compute_prototypes(train_loader, device)
    total_energy, per_cell_energy = compute_voronoi_energy(features, labels, prototypes)
    norm_energy, _ = compute_normalized_voronoi_energy(features, labels, prototypes)
    per_class_std, overall_std = compute_per_class_compactness(features, labels, prototypes)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nStep 3.1 - Final Voronoi Energy: {total_energy:.2f}")
    print(f"  Normalized (per sample): {norm_energy:.4f}")
    print(f"  Overall compactness (std): {overall_std:.4f}")
    print(f"\n  Per-cell energy:")
    for i in range(10):
        print(f"    Cell {i}: {per_cell_energy.get(i, 0):.2f}")
    
    # Compute total reduction
    initial_energy = voronoi_energies[0] if voronoi_energies else total_energy
    energy_reduction = initial_energy - total_energy
    energy_reduction_pct = (energy_reduction / initial_energy) * 100 if initial_energy > 0 else 0
    
    print(f"\n  Energy reduction from epoch {epoch_list[0]} to {epoch_list[-1]}: {energy_reduction:.2f} ({energy_reduction_pct:.1f}%)")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'voronoi_energies': voronoi_energies,
        'normalized_energies': normalized_energies,
        'per_class_compactness': per_class_compactness,
        'energy_drops': energy_drops,
        'accuracy_gains': accuracy_gains,
        'epoch_list': epoch_list,
        'final_energy': total_energy,
        'final_normalized_energy': norm_energy,
        'final_compactness': overall_std,
        'final_accuracy': test_acc,
        'energy_reduction': energy_reduction,
        'energy_reduction_pct': energy_reduction_pct,
        'config': config
    }
    
    torch.save(results, f"{config['save_dir']}/results.pt")
    
    # Generate plots
    plot_step3_results(train_losses, train_accuracies, test_accuracies,
                       voronoi_energies, normalized_energies, 
                       per_class_compactness, epoch_list,
                       accuracy_gains, energy_drops,
                       config['save_dir'])
    
    plot_energy_per_cell_heatmap(features, labels, prototypes, config['save_dir'])
    
    print(f"\nResults saved to {config['save_dir']}/")
    print("Step 3 Complete!")
    
    return model, results


# ============================================================================
# 5. PLOTTING FUNCTIONS
# ============================================================================

def plot_step3_results(train_losses, train_accuracies, test_accuracies,
                       voronoi_energies, normalized_energies, 
                       per_class_compactness, epoch_list,
                       accuracy_gains, energy_drops, save_dir):
    """Create summary figure for Step 3"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(train_losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(train_accuracies, 'g-', label='Train', linewidth=2)
    axes[0, 1].plot(test_accuracies, 'r-', label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Step 3.1 - Voronoi Energy (KEY FIGURE)
    axes[0, 2].plot(epoch_list, voronoi_energies, 'purple', marker='o', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Voronoi Quantization Energy')
    axes[0, 2].set_title('Step 3.1: Voronoi Energy Over Training')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Normalized Energy (per sample)
    axes[1, 0].plot(epoch_list, normalized_energies, 'orange', marker='s', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Normalized Energy (per sample)')
    axes[1, 0].set_title('Step 3.1: Normalized Voronoi Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Per-class compactness (within-class std)
    axes[1, 1].plot(epoch_list, per_class_compactness, 'teal', marker='^', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Within-Class Std Dev')
    axes[1, 1].set_title('Class Compactness (lower = tighter clusters)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Step 3.2 - Energy Drop vs Accuracy Gain correlation
    # Remove first entry (zero)
    valid_energy_drops = [d for d in energy_drops if d > 0]
    valid_accuracy_gains = [accuracy_gains[i] for i, d in enumerate(energy_drops) if d > 0]
    
    if len(valid_energy_drops) > 1:
        axes[1, 2].scatter(valid_energy_drops, valid_accuracy_gains, s=100, alpha=0.7, c='darkred')
        axes[1, 2].set_xlabel('Energy Drop (previous - current)')
        axes[1, 2].set_ylabel('Accuracy Gain')
        axes[1, 2].set_title('Step 3.2: Energy Drop vs Accuracy Gain')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(valid_energy_drops, valid_accuracy_gains, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(valid_energy_drops), max(valid_energy_drops), 100)
        axes[1, 2].plot(x_trend, p(x_trend), '--', color='gray', alpha=0.7)
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        axes[1, 2].set_title('Step 3.2: Energy Drop vs Accuracy Gain')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/step3_summary.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/step3_summary.png")


def plot_energy_per_cell_heatmap(features, labels, prototypes, save_dir):
    """Create heatmap of energy contribution per cell"""
    
    device = prototypes.device
    features = features.to(device)
    labels = labels.to(device)
    prototypes = prototypes.to(device)
    
    cell_assignments = assign_voronoi_cells(features, prototypes, device=device)
    
    k = len(prototypes)
    per_cell_energy = {}
    per_cell_count = {}
    
    for i in range(k):
        mask = (cell_assignments == i)
        if mask.sum() > 0:
            cell_features = features[mask]
            energy = torch.sum(torch.norm(cell_features - prototypes[i], dim=1) ** 2)
            per_cell_energy[i] = energy.item()
            per_cell_count[i] = mask.sum().item()
    
    # Create bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cells = list(range(k))
    energies = [per_cell_energy.get(i, 0) for i in cells]
    counts = [per_cell_count.get(i, 0) for i in cells]
    
    # Bar chart: Energy per cell
    axes[0].bar(cells, energies, color='purple', alpha=0.7)
    axes[0].set_xlabel('Cell (Prototype) Index')
    axes[0].set_ylabel('Voronoi Energy Contribution')
    axes[0].set_title('Energy per Voronoi Cell')
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart: Samples per cell
    axes[1].bar(cells, counts, color='teal', alpha=0.7)
    axes[1].set_xlabel('Cell (Prototype) Index')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Samples per Voronoi Cell')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/energy_per_cell.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/energy_per_cell.png")


def plot_energy_landscape_2d(prototypes, features, labels, save_dir, epoch):
    """Optional: Visualize energy landscape in 2D"""
    if prototypes.shape[1] != 2:
        return
    
    device = prototypes.device
    features = features.to(device)
    prototypes = prototypes.to(device)
    
    # Create grid
    margin = 1.0
    features_np = features.cpu().numpy()
    x_min, x_max = features_np[:, 0].min() - margin, features_np[:, 0].max() + margin
    y_min, y_max = features_np[:, 1].min() - margin, features_np[:, 1].max() + margin
    
    resolution = 100
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Compute energy at each grid point (distance to nearest prototype)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    distances = torch.cdist(grid_tensor, prototypes)
    min_distances = torch.min(distances, dim=1)[0]
    energy_grid = min_distances.reshape(xx.shape).cpu().numpy() ** 2
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(xx, yy, energy_grid, levels=50, cmap='viridis')
    ax.scatter(features_np[:, 0], features_np[:, 1], c=labels.cpu().numpy(), 
               cmap='tab10', s=5, alpha=0.5)
    ax.scatter(prototypes.cpu().numpy()[:, 0], prototypes.cpu().numpy()[:, 1], 
               c='red', marker='X', s=100, edgecolors='black')
    plt.colorbar(contour, ax=ax, label='Energy (distance² to nearest prototype)')
    ax.set_title(f'Voronoi Energy Landscape (Epoch {epoch})')
    ax.set_xlabel('Feature Dimension 1')
    ax.set_ylabel('Feature Dimension 2')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/energy_landscape_epoch_{epoch:04d}.png', dpi=150)
    plt.close()


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 3: Voronoi Quantization Energy')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Feature space dimension (2 or 3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    
    args = parser.parse_args()
    
    model, results = run_step3_experiment(
        feature_dim=args.dim,
        epochs=args.epochs,
        use_full_dataset=args.full
    )
