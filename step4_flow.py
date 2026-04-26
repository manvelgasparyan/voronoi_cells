"""
Step 4: Validating Voronoi Flow Dynamics (Theorem 3)

This script:
- 4.1: Computes cosine similarity between loss gradient and Voronoi alignment gradient
- 4.2: Analyzes effect of classification confidence on gradient alignment

Theorem 3 states: Under gradient descent, feature dynamics follow a preconditioned
gradient flow of the Voronoi energy, with ∇_z L ≈ (1/τ)(1-p_y)∇_z E_align

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


def compute_gradient_vectors(features, labels, prototypes, tau=1.0, logits=None):
    """
    Compute both:
    1. True loss gradient: ∇_z L
    2. Voronoi alignment gradient: ∇_z E_align = (z_y - z)  [FIXED SIGN]
    
    Args:
        features: (N, m) tensor
        labels: (N,) tensor
        prototypes: (k, m) tensor
        tau: temperature parameter
        logits: (N, k) tensor (optional, recomputed if not provided)
    
    Returns:
        grad_loss: (N, m) tensor
        grad_align: (N, m) tensor
        probs: (N, k) tensor
        p_correct: (N,) tensor (softmax probability for true class)
    """
    device = features.device
    k = len(prototypes)
    N = features.shape[0]
    
    # Compute softmax probabilities from distance-based logits
    distances = torch.cdist(features, prototypes) ** 2
    logits_dist = -0.5 * distances / tau
    probs = torch.softmax(logits_dist, dim=1)
    
    # True loss gradient: ∇_z L = (1/τ)[(z - z_y) - Σ_j p_j (z - z_j)]
    grad_loss = torch.zeros_like(features)
    for i in range(k):
        indicator = (labels == i).float().unsqueeze(1)
        factor = (probs[:, i:i+1] - indicator) / tau
        diff = (features - prototypes[i].unsqueeze(0))  # (z - z_i)
        grad_loss += factor * diff
    
    # Voronoi alignment gradient: ∇_z E_align = (z_y - z)  [FLIPPED SIGN]
    # With this sign, we expect positive cosine similarity
    grad_align = torch.zeros_like(features)
    for i in range(k):
        mask = (labels == i)
        if mask.sum() > 0:
            grad_align[mask] = prototypes[i].unsqueeze(0) - features[mask]  # (z_y - z)
    
    # Probability of correct class
    p_correct = probs[torch.arange(N), labels]
    
    return grad_loss, grad_align, probs, p_correct


def compute_cosine_similarity(grad_loss, grad_align):
    """
    Compute cosine similarity between two gradient vectors per sample.
    
    Returns:
        similarities: (N,) tensor of cosine similarities
    """
    # Normalize
    norm_loss = torch.norm(grad_loss, dim=1)
    norm_align = torch.norm(grad_align, dim=1)
    
    # Avoid division by zero
    norm_loss = torch.clamp(norm_loss, min=1e-8)
    norm_align = torch.clamp(norm_align, min=1e-8)
    
    # Dot product
    dot_product = torch.sum(grad_loss * grad_align, dim=1)
    similarities = dot_product / (norm_loss * norm_align)
    
    return similarities


def compute_alignment_metrics(features, labels, prototypes, tau=1.0):
    """
    Compute all alignment metrics for Theorem 3 validation.
    
    Returns:
        cosine_sims: (N,) tensor
        p_correct: (N,) tensor
        grad_loss_norm: (N,) tensor
        grad_align_norm: (N,) tensor
        alignment_factor: (N,) tensor = (1-p_y)/tau
    """
    grad_loss, grad_align, probs, p_correct = compute_gradient_vectors(
        features, labels, prototypes, tau=tau
    )
    
    cosine_sims = compute_cosine_similarity(grad_loss, grad_align)
    grad_loss_norm = torch.norm(grad_loss, dim=1)
    grad_align_norm = torch.norm(grad_align, dim=1)
    alignment_factor = (1 - p_correct) / tau
    
    return {
        'cosine_sims': cosine_sims,
        'p_correct': p_correct,
        'grad_loss_norm': grad_loss_norm,
        'grad_align_norm': grad_align_norm,
        'alignment_factor': alignment_factor,
        'theoretical_ratio': alignment_factor / (1 - p_correct + 1e-8)
    }


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


@torch.no_grad()
def compute_alignment_on_dataset(model, dataloader, prototypes, device, tau=1.0):
    """Compute alignment metrics over entire dataset"""
    model.eval()
    
    all_cosine_sims = []
    all_p_correct = []
    all_grad_loss_norms = []
    all_grad_align_norms = []
    all_margins = []
    
    for images, labels in tqdm(dataloader, desc="Computing alignment", leave=False):
        images, labels = images.to(device), labels.to(device)
        _, features = model.forward(images, return_features=True)
        
        metrics = compute_alignment_metrics(features, labels, prototypes, tau=tau)
        
        all_cosine_sims.append(metrics['cosine_sims'].cpu())
        all_p_correct.append(metrics['p_correct'].cpu())
        all_grad_loss_norms.append(metrics['grad_loss_norm'].cpu())
        all_grad_align_norms.append(metrics['grad_align_norm'].cpu())
        
        # Compute margin (min distance to other class)
        distances = torch.cdist(features, prototypes) ** 2
        correct_dist = distances[torch.arange(len(labels)), labels]
        wrong_dists = distances.clone()
        wrong_dists[torch.arange(len(labels)), labels] = float('inf')
        min_wrong_dist = torch.min(wrong_dists, dim=1)[0]
        margins = min_wrong_dist - correct_dist
        all_margins.append(margins.cpu())
    
    return {
        'cosine_sims': torch.cat(all_cosine_sims),
        'p_correct': torch.cat(all_p_correct),
        'grad_loss_norm': torch.cat(all_grad_loss_norms),
        'grad_align_norm': torch.cat(all_grad_align_norms),
        'margins': torch.cat(all_margins)
    }


# ============================================================================
# 4. MAIN EXPERIMENT (STEP 4)
# ============================================================================

def run_step4_experiment(feature_dim=2, epochs=50, use_full_dataset=False, tau=1.0):
    """
    Step 4: Validating Voronoi Flow Dynamics (Theorem 3)
    
    - 4.1: Cosine similarity between loss gradient and Voronoi alignment gradient
    - 4.2: Effect of classification confidence on gradient alignment
    """
    
    config = {
        'feature_dim': feature_dim,
        'num_classes': 10,
        'batch_size': 128,
        'epochs': epochs,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'tau': tau,
        'dataset': 'CIFAR-10',
        'save_dir': f'step4_results_flow_m{feature_dim}_tau{tau}',
        'eval_freq': 5
    }
    
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Feature dimension: {config['feature_dim']}")
    print(f"Temperature (τ): {config['tau']}")
    
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
        train_dataset = Subset(full_train, range(5000))
        test_dataset = Subset(full_test, range(1000))
        print(f"Subset: 5,000 train, 1,000 test")
    
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
    
    # Step 4 tracking
    mean_cosine_sims = []
    median_cosine_sims = []
    high_conf_cosine = []
    low_conf_cosine = []
    epoch_list = []
    
    # Per-sample tracking for detailed analysis
    all_cosine_by_confidence = []
    all_margins_by_cosine = []
    
    print("\n" + "="*60)
    print("Step 4.1: Cosine Similarity Between Gradients")
    print("Step 4.2: Effect of Classification Confidence on Alignment")
    print("Theorem 3: ∇_z L ≈ (1/τ)(1-p_y)∇_z E_align")
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
        
        # Periodic evaluation for alignment metrics
        if epoch % config['eval_freq'] == 0 or epoch == 1 or epoch == config['epochs']:
            # Get features and compute prototypes
            _, features, labels = evaluate(model, test_loader, device, compute_features=True)
            prototypes = model.compute_prototypes(train_loader, device)
            
            # Compute alignment metrics
            metrics = compute_alignment_on_dataset(model, test_loader, prototypes, device, tau=config['tau'])
            
            cosine_sims = metrics['cosine_sims']
            p_correct = metrics['p_correct']
            margins = metrics['margins']
            
            mean_cos = cosine_sims.mean().item()
            median_cos = cosine_sims.median().item()
            
            # Split by confidence
            high_conf_mask = p_correct > 0.9
            low_conf_mask = p_correct < 0.5
            
            high_conf_cos = cosine_sims[high_conf_mask].mean().item() if high_conf_mask.sum() > 0 else 0
            low_conf_cos = cosine_sims[low_conf_mask].mean().item() if low_conf_mask.sum() > 0 else 0
            
            mean_cosine_sims.append(mean_cos)
            median_cosine_sims.append(median_cos)
            high_conf_cosine.append(high_conf_cos)
            low_conf_cosine.append(low_conf_cos)
            epoch_list.append(epoch)
            
            # Store for detailed plots
            all_cosine_by_confidence.append((p_correct.numpy(), cosine_sims.numpy()))
            all_margins_by_cosine.append((margins.numpy(), cosine_sims.numpy()))
            
            print(f"\n[Epoch {epoch:03d}]")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            print(f"  Step 4.1 - Cosine Similarity: mean={mean_cos:.4f}, median={median_cos:.4f}")
            print(f"  Step 4.2 - High confidence (p>0.9): {high_conf_cos:.4f} | Low confidence (p<0.5): {low_conf_cos:.4f}")
            
            # Check Theorem 3 condition
            if mean_cos > 0.7:
                print(f"  ✓ Theorem 3 supported: Strong alignment (cos > 0.7)")
            elif mean_cos > 0.5:
                print(f"  ~ Partial alignment (cos = {mean_cos:.3f})")
            elif mean_cos > 0.3:
                print(f"  ~ Weak alignment (cos = {mean_cos:.3f})")
            else:
                print(f"  ✗ Poor alignment (cos = {mean_cos:.3f})")
        
        else:
            if epoch % 10 == 0:
                print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    # ========== Final Evaluation ==========
    print("\n" + "="*60)
    print("Final Results - Theorem 3 Validation")
    print("="*60)
    
    test_acc, features, labels = evaluate(model, test_loader, device, compute_features=True)
    prototypes = model.compute_prototypes(train_loader, device)
    metrics = compute_alignment_on_dataset(model, test_loader, prototypes, device, tau=config['tau'])
    
    final_mean_cos = metrics['cosine_sims'].mean().item()
    final_median_cos = metrics['cosine_sims'].median().item()
    high_conf_mask = metrics['p_correct'] > 0.9
    final_high_conf = metrics['cosine_sims'][high_conf_mask].mean().item() if high_conf_mask.sum() > 0 else 0
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"\nStep 4.1 - Final Cosine Similarity: {final_mean_cos:.4f} (mean), {final_median_cos:.4f} (median)")
    print(f"Step 4.2 - High confidence alignment: {final_high_conf:.4f}")
    
    # Theorem 3 verdict
    print("\n" + "="*60)
    print("Theorem 3 Verdict: ∇_z L ≈ (1/τ)(1-p_y)∇_z E_align")
    print("="*60)
    
    if final_mean_cos > 0.7:
        verdict = "SUPPORTED - Strong gradient alignment"
    elif final_mean_cos > 0.5:
        verdict = "PARTIALLY SUPPORTED - Moderate alignment"
    elif final_mean_cos > 0.3:
        verdict = "WEAKLY SUPPORTED - Low but positive alignment"
    else:
        verdict = "NOT SUPPORTED - Check sign or temperature"
    
    print(f"Verdict: {verdict}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'mean_cosine_sims': mean_cosine_sims,
        'median_cosine_sims': median_cosine_sims,
        'high_conf_cosine': high_conf_cosine,
        'low_conf_cosine': low_conf_cosine,
        'epoch_list': epoch_list,
        'final_cosine_sim': final_mean_cos,
        'final_accuracy': test_acc,
        'all_cosine_by_confidence': all_cosine_by_confidence,
        'all_margins_by_cosine': all_margins_by_cosine,
        'config': config
    }
    
    torch.save(results, f"{config['save_dir']}/results.pt")
    
    # Generate plots
    plot_step4_results(train_losses, train_accuracies, test_accuracies,
                       mean_cosine_sims, median_cosine_sims,
                       high_conf_cosine, low_conf_cosine, epoch_list,
                       config['save_dir'])
    
    plot_alignment_scatter(all_cosine_by_confidence, all_margins_by_cosine, config['save_dir'])
    
    print(f"\nResults saved to {config['save_dir']}/")
    print("Step 4 Complete!")
    
    return model, results


# ============================================================================
# 5. PLOTTING FUNCTIONS
# ============================================================================

def plot_step4_results(train_losses, train_accuracies, test_accuracies,
                       mean_cosine_sims, median_cosine_sims,
                       high_conf_cosine, low_conf_cosine, epoch_list, save_dir):
    """Create summary figure for Step 4"""
    
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
    
    # Plot 3: Step 4.1 - Cosine Similarity (KEY FIGURE)
    axes[0, 2].plot(epoch_list, mean_cosine_sims, 'purple', marker='o', linewidth=2, markersize=6, label='Mean')
    axes[0, 2].plot(epoch_list, median_cosine_sims, 'orange', marker='s', linewidth=2, markersize=6, label='Median')
    axes[0, 2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong alignment (0.7)')
    axes[0, 2].axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Moderate alignment (0.5)')
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.3, label='Zero')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Cosine Similarity')
    axes[0, 2].set_title('Step 4.1: Gradient Alignment (cos ∠(∇L, ∇E_align))')
    axes[0, 2].legend(loc='best')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Step 4.2 - Alignment by Confidence
    axes[1, 0].plot(epoch_list, high_conf_cosine, 'green', marker='^', linewidth=2, markersize=6, label='High confidence (p>0.9)')
    axes[1, 0].plot(epoch_list, low_conf_cosine, 'red', marker='v', linewidth=2, markersize=6, label='Low confidence (p<0.5)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_title('Step 4.2: Alignment by Classification Confidence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cosine vs Test Accuracy correlation (FIXED: match lengths)
    acc_at_cosine_epochs = [test_accuracies[e-1] for e in epoch_list]
    axes[1, 1].scatter(acc_at_cosine_epochs, mean_cosine_sims, s=100, alpha=0.7, c='darkorange')
    axes[1, 1].set_xlabel('Test Accuracy')
    axes[1, 1].set_ylabel('Mean Cosine Similarity')
    axes[1, 1].set_title('Alignment vs Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Final cosine similarity summary
    axes[1, 2].text(0.5, 0.5, f'Final mean cos = {mean_cosine_sims[-1]:.3f}\nFinal median cos = {median_cosine_sims[-1]:.3f}\n\nPositive = aligned\nNegative = opposite', 
                    ha='center', va='center', fontsize=12)
    axes[1, 2].set_title('Alignment Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/step4_summary.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/step4_summary.png")


def plot_alignment_scatter(all_cosine_by_confidence, all_margins_by_cosine, save_dir):
    """Create scatter plots for alignment analysis"""
    
    # Collect final epoch data
    final_confidence, final_cosine = all_cosine_by_confidence[-1]
    final_margins, _ = all_margins_by_cosine[-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter 1: Cosine similarity vs Confidence
    axes[0].scatter(final_confidence, final_cosine, alpha=0.3, s=10, c='blue')
    axes[0].set_xlabel('Classification Confidence (p_correct)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Gradient Alignment vs Model Confidence')
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(final_confidence, final_cosine)
    x_trend = np.linspace(0, 1, 100)
    axes[0].plot(x_trend, slope * x_trend + intercept, 'r--', 
                 label=f'Trend (r={r_value:.3f})')
    axes[0].legend()
    
    # Scatter 2: Cosine similarity vs Margin
    axes[1].scatter(final_margins, final_cosine, alpha=0.3, s=10, c='purple')
    axes[1].set_xlabel('Margin (min distance to wrong class - distance to correct)')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Gradient Alignment vs Decision Margin')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Decision boundary')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/alignment_scatter.png', dpi=150)
    plt.close()
    
    print(f"  Saved: {save_dir}/alignment_scatter.png")


# ============================================================================
# 6. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 4: Voronoi Flow Dynamics (Theorem 3)')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='Feature space dimension (2 or 3)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter (default: 1.0). Try 0.1 for sharper gradients')
    parser.add_argument('--full', action='store_true',
                        help='Use full CIFAR-10 dataset')
    
    args = parser.parse_args()
    
    model, results = run_step4_experiment(
        feature_dim=args.dim,
        epochs=args.epochs,
        use_full_dataset=args.full,
        tau=args.tau
    )
