"""
Step 2: Testing the Centroidal Voronoi Condition

This script:
- 2.1: Measures centroidal error per class over time
- 2.2: Verifies convergence of prototypes to cell centroids at stationarity

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
# 2. VORONOI UTILITIES (FIXED DEVICE MISMATCH)
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


def compute_cell_centroids(features, labels, prototypes):
    """
    Compute actual centroids of feature vectors within each Voronoi cell.
    
    Args:
        features: (N, m) tensor of feature vectors
        labels: (N,) tensor of true labels (not used for centroid, only for reference)
        prototypes: (k, m) tensor of class prototypes
    
    Returns:
        centroids: (k, m) tensor of actual cell centroids
        counts: (k,) tensor of number of points per cell
    """
    device = prototypes.device
    
    # Move everything to same device
    features = features.to(device)
    prototypes = prototypes.to(device)
    
    cell_assignments = assign_voronoi_cells(features, prototypes, device=device)
    
    k = len(prototypes)
    m = features.shape[1]
    centroids = torch.zeros(k, m, device=device)
    counts = torch.zeros(k, device=device)
    
    for i in range(k):
        mask = (cell_assignments == i)
        if mask.sum() > 0:
            centroids[i] = features[mask].mean(dim=0)
            counts[i] = mask.sum().float()
    
    return centroids.cpu(), counts.cpu()


def compute_centroidal_error(prototypes, centroids):
    """
    Compute the centroidal error per class.
    Error = ||prototype_i - centroid_i||^2
    
    Returns:
        per_class_error: (k,) tensor
        mean_error: float
    """
    prototypes = prototypes.cpu()
    centroids = centroids.cpu()
    
    per_class_error = torch.norm(prototypes - centroids, dim=1) ** 2
    mean_error = per_class_error.mean().item()
    
    return per_class_error.numpy(), mean_error


def compute_gradient_norm(model, dataloader, device):
    """Compute the norm of loss gradient w.r.t prototypes (for stationarity check)."""
    model.eval()
    
    k = model.classifier.out_features
    m = model.feature_layer.out_features
    
    grad_sum = torch.zeros(k, m, device=device)
    count = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        _, features = model.forward(images, return_features=True)
        
        tau = 1.0
        prototypes = model.prototypes.detach()
        
        distances = torch.cdist(features, prototypes) ** 2
        logits = -0.5 * distances / tau + model.classifier.bias.detach().unsqueeze(0)
        probs = torch.softmax(logits, dim=1)
        
        for i in range(k):
            indicator = (labels == i).float().unsqueeze(1)
            factor = (probs[:, i:i+1] - indicator) / tau
            diff = (prototypes[i].unsqueeze(0) - features)
            grad = (factor * diff).mean(dim=0)
            grad_sum[i] += grad.detach()
        count += 1
    
    grad_sum = grad_sum / count
    grad_norm_per_proto = torch.norm(grad_sum, dim=1).detach().cpu().numpy()
    total_grad_norm = torch.norm(grad_sum).detach().item()
    
    return grad_norm_per_proto, total_grad_norm


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
# 4. MAIN EXPERIMENT (STEP 2)
# ============================================================================

def run_step2_experiment(feature_dim=2, epochs=50, use_full_dataset=False):
    """
    Step 2: Testing the Centroidal Voronoi Condition
    
    - 2.1: Measures centroidal error per class over time
    - 2.2: Verifies convergence of prototypes to cell centroids at stationarity
    """
    
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dataset': 'CIFAR-10',
        'save_dir': f'step2_results_centroidal_m{feature_dim}',
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
    
    # Step 2.1 tracking
    centroidal_errors = []
    centroidal_errors_per_class = []
    epoch_list = []
    
    # Step 2.2 tracking
    grad_norms = []
    
    print("\n" + "="*60)
    print("Step 2.1: Measuring Centroidal Error per Class")
    print("Step 2.2: Verifying Convergence to Centroids at Stationarity")
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
        
        # Periodic evaluation
        if epoch % config['eval_freq'] == 0 or epoch == 1 or epoch == config['epochs']:
            # Get features
            _, features, labels = evaluate(model, test_loader, device, compute_features=True)
            
            # Compute prototypes from training data
            prototypes = model.compute_prototypes(train_loader, device)
            
            # Step 2.1: Compute centroids and error
            centroids, counts = compute_cell_centroids(features, labels, prototypes)
            per_class_error, mean_error = compute_centroidal_error(prototypes, centroids)
            centroidal_errors.append(mean_error)
            centroidal_errors_per_class.append(per_class_error)
            
            # Step 2.2: Compute gradient norm
            grad_norm_per_proto, total_grad_norm = compute_gradient_norm(model, train_loader, device)
            grad_norms.append(total_grad_norm)
            
            epoch_list.append(epoch)
            
            print(f"\n[Epoch {epoch:03d}]")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            print(f"  Step 2.1 - Centroidal Error: {mean_error:.6f}")
            print(f"    Per-class: {[round(e, 6) for e in per_class_error]}")
            print(f"  Step 2.2 - Gradient Norm (dL/dz): {total_grad_norm:.6f}")
            print(f"    Per-prototype: {[round(g, 6) for g in grad_norm_per_proto]}")
            
            if total_grad_norm < 0.01 and epoch > 20:
                print(f"  ✓ Stationarity reached (gradient norm < 0.01)")
        
        else:
            if epoch % 10 == 0:
                print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # ========== Final Evaluation ==========
    print("\n" + "="*60)
    print("Final Results - Stationarity Check (Theorem 5)")
    print("="*60)
    
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = model.compute_prototypes(train_loader, device)
    centroids, counts = compute_cell_centroids(features, labels, prototypes)
    per_class_error, mean_error = compute_centroidal_error(prototypes, centroids)
    grad_norm_per_proto, total_grad_norm = compute_gradient_norm(model, train_loader, device)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nStep 2.1 - Final Centroidal Error: {mean_error:.8f}")
    print(f"  Per-class errors:")
    for i, err in enumerate(per_class_error):
        print(f"    Class {i}: {err:.8f}")
    
    print(f"\nStep 2.2 - Final Gradient Norm (dL/dz): {total_grad_norm:.8f}")
    print(f"  Stationarity achieved: {'YES' if total_grad_norm < 0.01 else 'NO'}")
    
    # Theorem 5 verification
    print("\n" + "="*60)
    print("Theorem 5 Verification: Centroidal Voronoi Stationarity")
    print("="*60)
    
    condition_satisfied = mean_error < 0.1
    print(f"Centroidal condition: prototypes ≈ centroids? {'YES' if condition_satisfied else 'NO'}")
    print(f"  Mean squared distance: {mean_error:.6f}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'centroidal_errors': centroidal_errors,
        'centroidal_errors_per_class': centroidal_errors_per_class,
        'grad_norms': grad_norms,
        'epoch_list': epoch_list,
        'final_centroidal_error': mean_error,
        'final_grad_norm': total_grad_norm,
        'final_accuracy': test_acc,
        'centroidal_condition_met': condition_satisfied,
        'config': config
    }
    
    torch.save(results, f"{config['save_dir']}/results.pt")
    
    # Generate plots
    plot_step2_results(train_losses, train_accuracies, test_accuracies,
                       centroidal_errors, grad_norms, epoch_list,
                       config['save_dir'])
    
    plot_centroidal_heatmap(centroidal_errors_per_class, epoch_list, config['save_dir'])
    
    print(f"\nResults saved to {config['save_dir']}/")
    print("Step 2 Complete!")
    
    return model, results


# ============================================================================
# 5. PLOTTING FUNCTIONS
# ============================================================================

def plot_step2_results(train_losses, train_accuracies, test_accuracies,
                       centroidal_errors, grad_norms, epoch_list, save_dir):
    """Create summary figure for Step 2"""
    
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
    
    # Plot 3: Step 2.1 - Centroidal Error (KEY FIGURE)
    axes[0, 2].plot(epoch_list, centroidal_errors, 'purple', marker='o', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Centroidal Error ||z_i - centroid_i||²')
    axes[0, 2].set_title('Step 2.1: Centroidal Error Over Training')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='0.01 threshold')
    axes[0, 2].legend()
    
    # Plot 4: Step 2.2 - Gradient Norm
    axes[1, 0].plot(epoch_list, grad_norms, 'orange', marker='s', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('||∇_Z L|| (Gradient Norm)')
    axes[1, 0].set_title('Step 2.2: Gradient Norm (Stationarity Check)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Stationarity threshold')
    axes[1, 0].legend()
    
    # Plot 5: Centroidal Error vs Accuracy
    acc_at_epochs = [test_accuracies[e-1] for e in epoch_list]
    axes[1, 1].scatter(acc_at_epochs, centroidal_errors, s=100, alpha=0.7, c='darkorange')
    axes[1, 1].set_xlabel('Test Accuracy')
    axes[1, 1].set_ylabel('Centroidal Error')
    axes[1, 1].set_title('Centroidal Error vs Accuracy')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Gradient Norm vs Centroidal Error
    axes[1, 2].scatter(centroidal_errors, grad_norms, s=100, alpha=0.7, c='teal')
    axes[1, 2].set_xlabel('Centroidal Error')
    axes[1, 2].set_ylabel('Gradient Norm')
    axes[1, 2].set_title('Gradient Norm vs Centroidal Error')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/step2_summary.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/step2_summary.png")


def plot_centroidal_heatmap(centroidal_errors_per_class, epoch_list, save_dir):
    """Create heatmap of centroidal error per class over time"""
    
    errors = np.array(centroidal_errors_per_class)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(errors.T, aspect='auto', cmap='viridis_r', origin='lower')
    ax.set_xlabel('Evaluation (every 5 epochs)')
    ax.set_ylabel('Class')
    ax.set_title('Centroidal Error Per Class Over Training')
    ax.set_xticks(range(len(epoch_list)))
    ax.set_xticklabels(epoch_list)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10))
    
    plt.colorbar(im, ax=ax, label='||z_i - centroid_i||²')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/centroidal_error_heatmap.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/centroidal_error_heatmap.png")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 2: Centroidal Voronoi Condition')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Feature space dimension (2 or 3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    
    args = parser.parse_args()
    
    model, results = run_step2_experiment(
        feature_dim=args.dim,
        epochs=args.epochs,
        use_full_dataset=args.full
    )
