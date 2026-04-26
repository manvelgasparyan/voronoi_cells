"""
Step 5: Fixed-Point Analysis of Feature-Prototype Dynamics (Theorem 5)

This script verifies:
- 5.1: Stationarity condition: ∇_Z L = 0 at convergence
- 5.2: Prototypes coincide with cell centroids: z_i = centroid(V_i)

Theorem 5 states: At a stationary point of gradient flow with perfect classification,
the induced feature distribution and prototypes satisfy the centroidal Voronoi condition.

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


def compute_cell_centroids(features, prototypes, device=None):
    """
    Compute actual centroids of feature vectors within each Voronoi cell.
    
    Args:
        features: (N, m) tensor of feature vectors
        prototypes: (k, m) tensor of class prototypes
    
    Returns:
        centroids: (k, m) tensor of actual cell centroids
        counts: (k,) tensor of number of points per cell
    """
    if device is None:
        device = prototypes.device
    
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
    
    return centroids, counts


def compute_centroidal_error(prototypes, centroids):
    """
    Compute the centroidal error per class.
    Error = ||prototype_i - centroid_i||²
    
    Returns:
        per_class_error: (k,) array
        mean_error: float
        max_error: float
        relative_error: float (error / norm²)
    """
    prototypes = prototypes.cpu()
    centroids = centroids.cpu()
    
    proto_norms = torch.norm(prototypes, dim=1) ** 2
    per_class_error = torch.norm(prototypes - centroids, dim=1) ** 2
    
    mean_error = per_class_error.mean().item()
    max_error = per_class_error.max().item()
    
    # Relative error (avoid division by zero)
    relative_error = (per_class_error / (proto_norms + 1e-8)).mean().item()
    
    return per_class_error.numpy(), mean_error, max_error, relative_error


def compute_prototype_gradient_norm(model, dataloader, device):
    """
    Compute the norm of loss gradient w.r.t prototypes (∇_Z L).
    At stationarity (Theorem 5), this should be close to 0.
    
    Returns:
        grad_norm_per_prototype: array of norms per class
        total_grad_norm: float (Frobenius norm of all gradients)
    """
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
    grad_norm_per_proto = torch.norm(grad_sum, dim=1).cpu().numpy()
    total_grad_norm = torch.norm(grad_sum).detach().item()
    
    return grad_norm_per_proto, total_grad_norm


def compute_classification_confidence(model, dataloader, device):
    """
    Compute average softmax probability for correct class.
    Perfect classification (p_y = 1) is assumed in Theorem 5.
    """
    model.eval()
    all_p_correct = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            p_correct = probs[torch.arange(len(labels)), labels]
            all_p_correct.append(p_correct.cpu())
    
    all_p_correct = torch.cat(all_p_correct)
    return all_p_correct.mean().item(), all_p_correct.std().item()


def check_perfect_classification(model, dataloader, device, threshold=0.99):
    """
    Check if model achieves near-perfect classification (p_y > threshold).
    Theorem 5 assumes perfect classification.
    """
    mean_conf, std_conf = compute_classification_confidence(model, dataloader, device)
    is_perfect = mean_conf > threshold
    return is_perfect, mean_conf


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
# 4. MAIN EXPERIMENT (STEP 5)
# ============================================================================

def run_step5_experiment(feature_dim=2, epochs=100, use_full_dataset=False):
    """
    Step 5: Fixed-Point Analysis (Theorem 5)
    
    - 5.1: Verify stationarity: ∇_Z L = 0 at convergence
    - 5.2: Verify centroidal condition: z_i = centroid(V_i)
    """
    
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'dataset': 'CIFAR-10',
        'save_dir': f'step5_results_fixedpoint_m{feature_dim}',
        'eval_freq': 10,
        'stationarity_threshold': 0.01,  # ∇_Z L < 0.01
        'centroidal_threshold': 0.1,     # mean error < 0.1
        'perfect_classification_threshold': 0.95
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Feature dimension: {config['feature_dim']}")
    print(f"Epochs: {config['epochs']}")
    
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
    
    # Step 5 tracking
    grad_norms = []           # ||∇_Z L||
    centroidal_errors = []    # mean ||z_i - centroid_i||²
    relative_errors = []      # relative centroidal error
    class_confidences = []    # mean p_y
    epoch_list = []
    
    # Per-class tracking for final analysis
    final_per_class_grad_norms = None
    final_per_class_centroidal_errors = None
    
    print("\n" + "="*60)
    print("Step 5.1: Stationarity Condition (∇_Z L = 0)")
    print("Step 5.2: Centroidal Condition (z_i = centroid(V_i))")
    print("Theorem 5: Stationary points satisfy centroidal Voronoi condition")
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
        
        # Periodic evaluation for fixed-point metrics
        if epoch % config['eval_freq'] == 0 or epoch == 1 or epoch == config['epochs']:
            # Get features and prototypes
            _, features, labels = evaluate(model, test_loader, device, compute_features=True)
            prototypes = model.compute_prototypes(train_loader, device)
            
            # Step 5.1: Compute gradient norm (stationarity)
            grad_norm_per_proto, total_grad_norm = compute_prototype_gradient_norm(
                model, train_loader, device
            )
            
            # Step 5.2: Compute centroidal error
            centroids, counts = compute_cell_centroids(features, prototypes, device)
            per_class_error, mean_error, max_error, rel_error = compute_centroidal_error(
                prototypes, centroids
            )
            
            # Classification confidence (how close to perfect classification)
            mean_conf, std_conf = compute_classification_confidence(model, test_loader, device)
            
            grad_norms.append(total_grad_norm)
            centroidal_errors.append(mean_error)
            relative_errors.append(rel_error)
            class_confidences.append(mean_conf)
            epoch_list.append(epoch)
            
            # Store final per-class values
            if epoch == config['epochs']:
                final_per_class_grad_norms = grad_norm_per_proto
                final_per_class_centroidal_errors = per_class_error
            
            # Check Theorem 5 conditions
            stationarity_achieved = total_grad_norm < config['stationarity_threshold']
            centroidal_achieved = mean_error < config['centroidal_threshold']
            perfect_classification = mean_conf > config['perfect_classification_threshold']
            
            print(f"\n[Epoch {epoch:03d}]")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            print(f"  Step 5.1 - ∇_Z L norm: {total_grad_norm:.6f} {'✓' if stationarity_achieved else '✗'}")
            print(f"  Step 5.2 - Centroidal error: {mean_error:.6f} {'✓' if centroidal_achieved else '✗'}")
            print(f"  Classification confidence (p_y): {mean_conf:.4f} ± {std_conf:.4f}")
            print(f"  Relative error (error / ||z_i||²): {rel_error:.6f}")
            
            if stationarity_achieved and centroidal_achieved:
                print(f"  ✓ Theorem 5 conditions SATISFIED at epoch {epoch}")
            elif stationarity_achieved:
                print(f"  ~ Stationarity achieved, centroidal condition pending")
            elif centroidal_achieved:
                print(f"  ~ Centroidal condition achieved, stationarity pending")
        
        else:
            if epoch % 20 == 0:
                print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # ========== Final Evaluation ==========
    print("\n" + "="*60)
    print("Final Results - Theorem 5 Verification")
    print("="*60)
    
    # Final feature extraction
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = model.compute_prototypes(train_loader, device)
    centroids, counts = compute_cell_centroids(features, prototypes, device)
    per_class_error, final_mean_error, final_max_error, final_rel_error = compute_centroidal_error(prototypes, centroids)
    grad_norm_per_proto, final_grad_norm = compute_prototype_gradient_norm(model, train_loader, device)
    final_mean_conf, final_std_conf = compute_classification_confidence(model, test_loader, device)
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nStep 5.1 - Final ∇_Z L norm: {final_grad_norm:.8f}")
    print(f"  Per-class gradient norms: {[round(g, 6) for g in grad_norm_per_proto]}")
    print(f"\nStep 5.2 - Final Centroidal Error: {final_mean_error:.8f}")
    print(f"  Max per-class error: {final_max_error:.8f}")
    print(f"  Relative error: {final_rel_error:.8f}")
    print(f"  Per-class errors: {[round(e, 6) for e in per_class_error]}")
    print(f"\nClassification confidence: {final_mean_conf:.4f} ± {final_std_conf:.4f}")
    
    # Theorem 5 verdict
    print("\n" + "="*60)
    print("Theorem 5 Verdict: Stationary points satisfy centroidal Voronoi condition")
    print("="*60)
    
    stationarity_ok = final_grad_norm < config['stationarity_threshold']
    centroidal_ok = final_mean_error < config['centroidal_threshold']
    confidence_ok = final_mean_conf > config['perfect_classification_threshold']
    
    print(f"  Stationarity (∇_Z L < 0.01): {'✓' if stationarity_ok else '✗'} ({final_grad_norm:.6f})")
    print(f"  Centroidal condition (error < 0.1): {'✓' if centroidal_ok else '✗'} ({final_mean_error:.6f})")
    print(f"  Near-perfect classification (p_y > 0.95): {'✓' if confidence_ok else '✗'} ({final_mean_conf:.4f})")
    
    if stationarity_ok and centroidal_ok:
        verdict = "CONFIRMED - Network reached centroidal Voronoi configuration"
    elif stationarity_ok:
        verdict = "PARTIALLY CONFIRMED - Stationary but not centroidal (needs more training)"
    elif centroidal_ok:
        verdict = "PARTIALLY CONFIRMED - Centroidal but not stationary"
    else:
        verdict = "INCONCLUSIVE - Train longer or increase feature dimension"
    
    print(f"\nVerdict: {verdict}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'grad_norms': grad_norms,
        'centroidal_errors': centroidal_errors,
        'relative_errors': relative_errors,
        'class_confidences': class_confidences,
        'epoch_list': epoch_list,
        'final_grad_norm': final_grad_norm,
        'final_centroidal_error': final_mean_error,
        'final_relative_error': final_rel_error,
        'final_confidence': final_mean_conf,
        'per_class_grad_norms': grad_norm_per_proto.tolist() if hasattr(grad_norm_per_proto, 'tolist') else grad_norm_per_proto,
        'per_class_centroidal_errors': per_class_error.tolist(),
        'stationarity_achieved': stationarity_ok,
        'centroidal_achieved': centroidal_ok,
        'verdict': verdict,
        'config': config
    }
    
    torch.save(results, f"{config['save_dir']}/results.pt")
    
    # Generate plots
    plot_step5_results(train_losses, train_accuracies, test_accuracies,
                       grad_norms, centroidal_errors, class_confidences, epoch_list,
                       config['save_dir'])
    
    plot_per_class_analysis(per_class_error, grad_norm_per_proto, counts.cpu().numpy(), 
                            prototypes.cpu().numpy(), centroids.cpu().numpy(),
                            config['save_dir'])
    
    plot_convergence_comparison(centroidal_errors, grad_norms, class_confidences, epoch_list,
                                 config['save_dir'])
    
    print(f"\nResults saved to {config['save_dir']}/")
    print("Step 5 Complete!")
    
    return model, results


# ============================================================================
# 5. PLOTTING FUNCTIONS
# ============================================================================

def plot_step5_results(train_losses, train_accuracies, test_accuracies,
                       grad_norms, centroidal_errors, class_confidences, epoch_list, save_dir):
    """Create summary figure for Step 5"""
    
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
    
    # Plot 3: Step 5.1 - Gradient Norm (Stationarity)
    axes[0, 2].plot(epoch_list, grad_norms, 'purple', marker='o', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('||∇_Z L|| (Gradient Norm)')
    axes[0, 2].set_title('Step 5.1: Stationarity (∇_Z L → 0)')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Stationarity threshold')
    axes[0, 2].legend()
    
    # Plot 4: Step 5.2 - Centroidal Error
    axes[1, 0].plot(epoch_list, centroidal_errors, 'orange', marker='s', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean ||z_i - centroid_i||²')
    axes[1, 0].set_title('Step 5.2: Centroidal Error')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Centroidal threshold')
    axes[1, 0].legend()
    
    # Plot 5: Classification Confidence
    axes[1, 1].plot(epoch_list, class_confidences, 'teal', marker='^', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Mean p_y (confidence)')
    axes[1, 1].set_title('Classification Confidence (p_y → 1)')
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Perfect classification threshold')
    axes[1, 1].legend()
    
    # Plot 6: Gradient Norm vs Centroidal Error correlation
    axes[1, 2].scatter(centroidal_errors, grad_norms, s=100, alpha=0.7, c='darkred')
    axes[1, 2].set_xlabel('Centroidal Error')
    axes[1, 2].set_ylabel('Gradient Norm')
    axes[1, 2].set_title('Correlation: Stationarity vs Centroidal')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/step5_summary.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/step5_summary.png")


def plot_per_class_analysis(per_class_error, grad_norm_per_proto, counts, prototypes, centroids, save_dir):
    """Create per-class bar charts"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = range(len(per_class_error))
    
    # Bar chart: Centroidal error per class
    axes[0].bar(classes, per_class_error, color='orange', alpha=0.7)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('||z_i - centroid_i||²')
    axes[0].set_title('Centroidal Error Per Class')
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart: Gradient norm per class
    axes[1].bar(classes, grad_norm_per_proto, color='purple', alpha=0.7)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('||∇_Z_i L||')
    axes[1].set_title('Gradient Norm Per Class (Stationarity)')
    axes[1].grid(True, alpha=0.3)
    
    # Bar chart: Samples per Voronoi cell
    axes[2].bar(classes, counts, color='teal', alpha=0.7)
    axes[2].set_xlabel('Voronoi Cell')
    axes[2].set_ylabel('Number of Samples')
    axes[2].set_title('Samples per Voronoi Cell')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_class_analysis.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/per_class_analysis.png")
    
    # Also create prototype vs centroid visualization for 2D
    if prototypes.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot centroids and prototypes
        ax.scatter(prototypes[:, 0], prototypes[:, 1], c='red', marker='X', s=200, 
                   edgecolors='black', linewidth=2, label='Prototypes (z_i)')
        ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='o', s=100,
                   edgecolors='black', linewidth=1, label='Cell Centroids')
        
        # Draw lines connecting prototypes to centroids
        for i in range(len(prototypes)):
            ax.plot([prototypes[i, 0], centroids[i, 0]], 
                   [prototypes[i, 1], centroids[i, 1]], 
                   'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Feature Dimension 1')
        ax.set_ylabel('Feature Dimension 2')
        ax.set_title('Prototypes vs Cell Centroids (Closer = Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prototypes_vs_centroids.png', dpi=150)
        plt.close()
        
        print(f"  Saved: {save_dir}/prototypes_vs_centroids.png")


def plot_convergence_comparison(centroidal_errors, grad_norms, class_confidences, epoch_list, save_dir):
    """Plot all three convergence metrics together"""
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot centroidal error and gradient norm on primary y-axis
    ax1.plot(epoch_list, centroidal_errors, 'orange', marker='s', linewidth=2, 
             markersize=6, label='Centroidal Error')
    ax1.plot(epoch_list, grad_norms, 'purple', marker='o', linewidth=2, 
             markersize=6, label='Gradient Norm')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error / Norm (log scale)', color='black')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Plot confidence on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(epoch_list, class_confidences, 'teal', marker='^', linewidth=2, 
             markersize=6, label='Classification Confidence')
    ax2.set_ylabel('Confidence (p_y)', color='teal')
    ax2.tick_params(axis='y', labelcolor='teal')
    
    # Add thresholds
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Centroidal threshold')
    ax1.axhline(y=0.01, color='purple', linestyle='--', alpha=0.5, label='Stationarity threshold')
    ax2.axhline(y=0.95, color='teal', linestyle='--', alpha=0.5, label='Perfect classification threshold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax1.set_title('Theorem 5 Convergence: All Metrics Together')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_comparison.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/convergence_comparison.png")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 5: Fixed-Point Analysis (Theorem 5)')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3, 4, 8],
                        help='Feature space dimension (2, 3, 4, or 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (more = better convergence)')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    
    args = parser.parse_args()
    
    model, results = run_step5_experiment(
        feature_dim=args.dim,
        epochs=args.epochs,
        use_full_dataset=args.full
    )
