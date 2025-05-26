import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
import seaborn as sns

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        print(f"The current model save path is: {path + '/' + 'checkpoint.pth'}")  # Add this line to print the save path
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def load_content(args):
    dataset_name = os.path.basename(os.path.normpath(args.root_path))
    print(dataset_name)
    if 'ETT' in args.data:
        file = 'ETT'
    elif dataset_name == 'traffic':
        file = 'Traffic'
    elif dataset_name == 'electricity':
        file = 'ECL'
    elif dataset_name == 'weather':
        file = 'Weather'
    elif dataset_name == 'illness':
        file = 'ILI'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


def visualize_embeddings_difference(patch_features, fused_features, save_path='embedding_difference.png'):
    """
    Visualize the difference between patch_features and fused_features.
    """
    fused_mean, fused_var = fused_features.mean(), fused_features.var()
    patch_mean, patch_var = patch_features.mean(), patch_features.var()
    print(f"Fused Features - Mean: {fused_mean}, Variance: {fused_var}")
    print(f"Patch Features - Mean: {patch_mean}, Variance: {patch_var}")
    cosine_sim = torch.nn.functional.cosine_similarity(fused_features, patch_features, dim=-1)
    print(f"Cosine Similarity: {cosine_sim.mean()}")
            

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14  # Set global font size

def visualize_embeddings(patch_features, fused_features, save_path='embedding_distribution.png'):
    """
    Visualize the spatial distribution of patch_embedding and fused_embedding.
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(patch_features, torch.Tensor):
        patch_features = torch.tensor(patch_features)
    if not isinstance(fused_features, torch.Tensor):
        fused_features = torch.tensor(fused_features)

    # Move tensors from GPU to CPU and convert to NumPy arrays
    patch_embedding = patch_features.detach().cpu().numpy()  # [B * n_vars, d_model]
    fused_embedding = fused_features.detach().cpu().numpy()  # [B * n_vars, d_model]

    # Randomly sample points
    num_samples = min(1000, patch_embedding.shape[0])
    patch_embedding = patch_embedding[np.random.choice(patch_embedding.shape[0], num_samples, replace=False)]
    fused_embedding = fused_embedding[np.random.choice(fused_embedding.shape[0], num_samples, replace=False)]

    # Reduce dimensions using UMAP
    umap = UMAP(n_components=2, random_state=42)
    patch_embedding_2d = umap.fit_transform(patch_embedding)
    fused_embedding_2d = umap.fit_transform(fused_embedding)

    # Create visualization
    plt.figure(figsize=(8, 8))
    sns.set_palette("husl")  # Use seaborn color palette
    plt.scatter(patch_embedding_2d[:, 0], patch_embedding_2d[:, 1], c='blue', label='Temporal Embedding', alpha=0.7, s=300, edgecolor='k', linewidth=0.8)
    plt.scatter(fused_embedding_2d[:, 0], fused_embedding_2d[:, 1], c='red', label='Multimodal Embedding', alpha=0.7, s=300, edgecolor='k', linewidth=0.8)
    
    # Add legend and title
    plt.legend(fontsize=18, frameon=True, framealpha=0.8, loc='upper right')
    plt.title('2D UMAP Visualization of Temporal and Multimodal Embeddings', fontsize=21, pad=20, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=20, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=20, fontweight='bold')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Optimize layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_gate_weights(gate_weights, save_path='gate_weights_distribution.png'):
    """
    Visualize the distribution of gate weights.

    Args:
        gate_weights (torch.Tensor): Gate weights with shape [B * n_vars, 2].
        save_path (str): Path to save visualization, defaults to 'gate_weights_distribution.png'.
    """
    # Extract weights for fused_features and patch_features
    fused_weights = gate_weights[:, 0].detach().cpu().numpy().flatten()  # Extract fused_features weights
    patch_weights = gate_weights[:, 1].detach().cpu().numpy().flatten()  # Extract patch_features weights

    # Create visualization
    plt.figure(figsize=(8, 8))
    
    # Use consistent colors with visualize_embeddings (red and blue)
    plt.hist(fused_weights, bins=20, alpha=0.6, label='Multimodal Features Weights', color='red', edgecolor='black', linewidth=1.2)
    plt.hist(patch_weights, bins=20, alpha=0.6, label='Temporal Features Weights', color='blue', edgecolor='black', linewidth=1.2)
    
    # Add title and labels
    plt.xlabel('Gate Weight Value', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.title('Distribution of Multimodal and Temporal Gate Weights', fontsize=21, pad=20, fontweight='bold')
    
    # Add legend
    plt.legend(fontsize=18, frameon=True, framealpha=0.8, loc='upper right')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Optimize layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_three_modal_embeddings_difference(memory_features, vision_features, text_features, save_path='embedding_difference.png'):
    """
    Visualize the difference between patch_features and fused_features.
    """
    memory_mean, memory_var = memory_features.mean(), memory_features.var()
    vision_mean, vision_var = vision_features.mean(), vision_features.var()
    text_mean, text_var = text_features.mean(), text_features.var()
    print(f"Memory Features - Mean: {memory_mean}, Variance: {memory_var}")
    print(f"Vision Features - Mean: {vision_mean}, Variance: {vision_var}")
    print(f"Text Features - Mean: {text_mean}, Variance: {text_var}")
    cosine_sim = torch.nn.functional.cosine_similarity(memory_features, vision_features, dim=-1)
    print(f"Cosine Similarity between Memory and Vision: {cosine_sim.mean()}")
    cosine_sim = torch.nn.functional.cosine_similarity(memory_features, text_features, dim=-1)
    print(f"Cosine Similarity between Memory and Text: {cosine_sim.mean()}")
    cosine_sim = torch.nn.functional.cosine_similarity(vision_features, text_features, dim=-1)
    print(f"Cosine Similarity between Vision and Text: {cosine_sim.mean()}")

def visualize_four_modal_features(memory_features, vision_features, text_features, fused_features, save_path='four_modal_features.png'):
    """
    Visualize the spatial distribution of three different modal features using UMAP.
    
    Args:
        memory_features (torch.Tensor): Temporal features [B, n_vars, d_model]
        vision_features (torch.Tensor): Visual features [B, n_vars, d_model]
        text_features (torch.Tensor): Text features [B, n_vars, d_model]
        save_path (str): Path to save the visualization
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(memory_features, torch.Tensor):
        memory_features = torch.tensor(memory_features)
    if not isinstance(vision_features, torch.Tensor):
        vision_features = torch.tensor(vision_features)
    if not isinstance(text_features, torch.Tensor):
        text_features = torch.tensor(text_features)
    if not isinstance(fused_features, torch.Tensor):
        fused_features = torch.tensor(fused_features)

    # Move tensors from GPU to CPU and convert to NumPy arrays
    memory_embedding = memory_features.detach().cpu().numpy()
    vision_embedding = vision_features.detach().cpu().numpy()
    text_embedding = text_features.detach().cpu().numpy()
    fused_embedding = fused_features.detach().cpu().numpy()
    
    # Reshape 3D tensors to 2D for UMAP
    B, n_vars, d_model = memory_embedding.shape
    memory_embedding = memory_embedding.reshape(-1, d_model)  # [B * n_vars, d_model]
    vision_embedding = vision_embedding.reshape(-1, d_model)  # [B * n_vars, d_model]
    text_embedding = text_embedding.reshape(-1, d_model)      # [B * n_vars, d_model]
    fused_embedding = fused_embedding.reshape(-1, d_model)    # [B * n_vars, d_model]
    
    # Randomly sample points for better visualization
    num_samples = min(1000, memory_embedding.shape[0])
    memory_embedding = memory_embedding[np.random.choice(memory_embedding.shape[0], num_samples, replace=False)]
    vision_embedding = vision_embedding[np.random.choice(vision_embedding.shape[0], num_samples, replace=False)]
    text_embedding = text_embedding[np.random.choice(text_embedding.shape[0], num_samples, replace=False)]
    fused_embedding = fused_embedding[np.random.choice(fused_embedding.shape[0], num_samples, replace=False)]
    
    # Reduce dimensions using UMAP
    umap = UMAP(n_components=2, random_state=42)
    memory_embedding_2d = umap.fit_transform(memory_embedding)
    vision_embedding_2d = umap.fit_transform(vision_embedding)
    text_embedding_2d = umap.fit_transform(text_embedding)
    fused_embedding_2d = umap.fit_transform(fused_embedding)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    sns.set_palette("husl")  # Use seaborn color palette
    
    # Plot each modal feature with different colors and markers
    plt.scatter(memory_embedding_2d[:, 0], memory_embedding_2d[:, 1], 
               c='blue', label='Temporal Features', alpha=0.7, s=300, 
               edgecolor='k', linewidth=0.8, marker='o')
    plt.scatter(vision_embedding_2d[:, 0], vision_embedding_2d[:, 1], 
               c='red', label='Visual Features', alpha=0.7, s=300, 
               edgecolor='k', linewidth=0.8, marker='s')
    plt.scatter(text_embedding_2d[:, 0], text_embedding_2d[:, 1], 
               c='green', label='Text Features', alpha=0.7, s=300, 
               edgecolor='k', linewidth=0.8, marker='^')
    plt.scatter(fused_embedding_2d[:, 0], fused_embedding_2d[:, 1], 
               c='purple', label='Fused Features', alpha=0.7, s=300, 
               edgecolor='k', linewidth=0.8, marker='*')
    
    # Add legend and title
    plt.legend(fontsize=18, frameon=True, framealpha=0.8, loc='upper right')
    plt.title('2D UMAP Visualization of Four Modal Features', fontsize=21, pad=20, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=20, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=20, fontweight='bold')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Optimize layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_three_gate_weights(gate_weights, save_path='three_modal_gate_weights.png'):
    """
    Visualize the distribution of three modal gate weights.

    Args:
        gate_weights (torch.Tensor): Gate weights with shape [B, n_vars, 3].
        save_path (str): Path to save visualization, defaults to 'three_modal_gate_weights.png'.
    """
    # Extract weights for each modality
    temporal_weights = gate_weights[:, :, 0].detach().cpu().numpy().flatten()  # Temporal features weights
    visual_weights = gate_weights[:, :, 1].detach().cpu().numpy().flatten()    # Visual features weights
    text_weights = gate_weights[:, :, 2].detach().cpu().numpy().flatten()      # Text features weights

    # Create visualization
    plt.figure(figsize=(10, 10))
    
    # Plot histograms for each modal weight
    plt.hist(temporal_weights, bins=20, alpha=0.6, label='Temporal Features Weights', 
             color='blue', edgecolor='black', linewidth=1.2)
    plt.hist(visual_weights, bins=20, alpha=0.6, label='Visual Features Weights', 
             color='red', edgecolor='black', linewidth=1.2)
    plt.hist(text_weights, bins=20, alpha=0.6, label='Text Features Weights', 
             color='green', edgecolor='black', linewidth=1.2)
    
    # Add title and labels
    plt.xlabel('Gate Weight Value', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.title('Distribution of Three Modal Gate Weights', fontsize=21, pad=20, fontweight='bold')
    
    # Add legend
    plt.legend(fontsize=18, frameon=True, framealpha=0.8, loc='upper right')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Optimize layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def check_numerical_stability(tensor, name=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or Inf values")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean()}, Std: {tensor.std()}")
        print(f"Min: {tensor.min()}, Max: {tensor.max()}")
        return False
    return True