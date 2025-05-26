import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.preprocessing import StandardScaler
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
import torch.nn as nn

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants for number of samples
NUM_TS_SAMPLES = 200  # Number of samples for time series datasets (increased from 60)
NUM_COCO_SAMPLES = 400  # Number of samples for COCO dataset (increased from 200)
            
# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14  # Set global font size


def process_dataset(dataset, window_size=96, num_samples=NUM_TS_SAMPLES):
    """Process a dataset and return its features."""
    # Determine dataset path and files
    if dataset.startswith('ETT'):
        dataset_path = os.path.join('dataset', 'ETT-small')
        csv_file = f"{dataset}.csv"
        csv_path = os.path.join(dataset_path, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} does not exist, skipping...")
            return None
            
        df = pd.read_csv(csv_path)
        csv_files = [csv_file]
    else:
        dataset_path = os.path.join('dataset', dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist, skipping...")
            return None
            
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {dataset_path}")
            return None
    
    # Process data from all CSV files
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(file_path)
        
        # Skip the date column and convert to tensor
        try:
            data = df.iloc[:, 1:].astype(float).values
        except:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[1:]
            if len(numeric_cols) == 0:
                print(f"No numeric columns found in {csv_file}, skipping...")
                continue
            data = df[numeric_cols].values
        
        # Check for NaN values
        if np.isnan(data).any():
            print(f"Warning: NaN values found in {csv_file}, filling with forward fill...")
            data = pd.DataFrame(data).fillna(method='ffill').values
        
        all_data.append(data)
    
    if not all_data:
        print(f"Warning: No data found for {dataset}, skipping...")
        return None
    
    # Concatenate all data
    all_data = np.concatenate(all_data, axis=0)
    
    # Global normalization to [0,255] range
    min_vals = all_data.min(axis=0, keepdims=True)
    max_vals = all_data.max(axis=0, keepdims=True)
    all_data = ((all_data - min_vals) / (max_vals - min_vals + 1e-8)) * 255  # Normalize to [0,255] range
    
    # Create sliding windows
    stride = window_size // 2
    n_windows = (all_data.shape[0] - window_size) // stride + 1
    
    print(f"Creating {n_windows} windows for {dataset}")
    
    windows = []
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        window = all_data[start_idx:end_idx]
        windows.append(window)
    
    # Sample windows
    num_samples = min(num_samples, len(windows))
    indices = np.random.choice(len(windows), num_samples, replace=False)
    sampled_windows = [windows[i] for i in indices]
    
    print(f"Sampled {num_samples} windows for {dataset}")
    
    # Convert to tensor and reshape to [B, seq_len, n_vars]
    sampled_data = torch.tensor(np.stack(sampled_windows), dtype=torch.float32)
    if len(sampled_data.shape) == 2:
        sampled_data = sampled_data.unsqueeze(-1)
    
    return sampled_data

def visualize_embeddings(all_features, feature_labels, save_path='embeddings_visualization.png', title='UMAP Visualization of Features'):
    """Visualize embeddings using UMAP."""
    # UMAP parameters
    umap = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=1.2,
        metric='cosine',
        random_state=42,
        spread=1.2,
        repulsion_strength=2
    )
    all_features_2d = umap.fit_transform(all_features)
    
    plt.figure(figsize=(12, 8))
    
    # Color scheme
    colors = {
        # COCO features with aesthetically pleasing colors
        'COCO-Pair': '#E74C3C',    # Soft Red
        'COCO-Image': '#2ECC71',   # Soft Green
        'COCO-Text': '#3498DB',    # Soft Blue
        
        # Time series datasets with muted colors
        'ETTh1': '#FFA500',        # Orange
        'ETTh2': '#FFD700',        # Gold
        'ETTm1': '#FF8C00',        # Dark Orange
        'ETTm2': '#FFB6C1',        # Light Pink
        'electricity': '#9370DB',  # Medium Purple
        'traffic': '#20B2AA',      # Light Sea Green
        'weather': '#8B4513',      # Saddle Brown
    }
    
    unique_labels = list(set(feature_labels))
    unique_labels.sort()
    
    # Plot with improved visualization and larger markers for COCO features
    for label in unique_labels:
        mask = [l == label for l in feature_labels]
        if label.startswith('COCO'):
            # COCO features with larger markers
            plt.scatter(
                all_features_2d[mask, 0],
                all_features_2d[mask, 1],
                c=colors.get(label, '#1f77b4'),
                label=label,
                alpha=0.8,  # Higher opacity for COCO features
                s=150,     # Even larger markers for COCO features
                edgecolors='black',  # Add black border
                linewidth=0.8
            )
        else:
            # Time series features with smaller markers
            plt.scatter(
                all_features_2d[mask, 0],
                all_features_2d[mask, 1],
                c=colors.get(label, '#1f77b4'),
                label=label,
                alpha=0.5,  # Slightly higher opacity for time series
                s=80,      # Larger markers for time series
                edgecolors='none'
            )
    
    # Plot settings
    plt.legend(
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        ncol=2,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        markerscale=1.5
    )
    
    # Enhanced title and labels with larger fonts
    plt.title(title, fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=16)
    plt.ylabel('UMAP Dimension 2', fontsize=16)
    
    # Enhanced grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with higher DPI and adjusted bbox
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        pad_inches=0.2
    )
    plt.close()
    
    # Print statistics
    print(f"\nFeature Statistics for {title}:")
    for label in unique_labels:
        mask = [l == label for l in feature_labels]
        num_samples = sum(mask)
        print(f"{label}: {num_samples} samples")

def visualize_ts_embeddings_with_coco_samples(save_path='ts_embeddings_with_coco_samples.png'):
    """Visualize time series embeddings with COCO samples."""
    # Load COCO samples
    coco_dir = "tutorial/visualization/coco_samples"
    image_dir = os.path.join(coco_dir, "images")
    caption_file = os.path.join(coco_dir, "filtered_captions.txt")
    
    # Load models
    CLIP_ARCH = 'openai/clip-vit-base-patch32'
    processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
    model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
    model.eval()
    
    # Load ViT and BERT models for separate embeddings
    from transformers import ViTModel, ViTFeatureExtractor, BertTokenizer, BertModel
    vit_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Model dimensions
    clip_dim = 512
    vit_dim = 768
    bert_dim = 768
    target_dim = 512
    
    # Projection layers
    clip_proj = nn.Linear(clip_dim, target_dim)
    vit_proj = nn.Linear(vit_dim, target_dim)
    bert_proj = nn.Linear(bert_dim, target_dim)
    
    # Set models to eval mode
    vit_model.eval()
    bert_model.eval()
    clip_proj.eval()
    vit_proj.eval()
    bert_proj.eval()
    
    # Initialize time series to image converter
    ts_to_image = LearnableTimeSeriesToImage(
        input_dim=3,
        hidden_dim=48,
        output_channels=3,
        image_size=224,
        periodicity=24
    )
    
    # List of datasets to process
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity', 'traffic']
    ts_features_dict = {}
    
    # Process time series data
    for dataset in datasets:
        sampled_data = process_dataset(dataset)
        if sampled_data is None:
            continue
            
        fused_features = []
        num_samples = sampled_data.shape[0]
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Processing sample {i+1}/{num_samples} for {dataset}")
            
            ts_sample = sampled_data[i:i+1]
            
            with torch.no_grad():
                image = ts_to_image(ts_sample)
                image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
                image_np = image[0].cpu().numpy().transpose(1, 2, 0)
                image_pil = Image.fromarray(image_np.astype(np.uint8))
            
            # Extract time series statistics
            ts_data = ts_sample[0]
            max_val = ts_data.max().item()
            min_val = ts_data.min().item()
            median_val = ts_data.median().item()
            
            trends = (ts_data[:, -1] - ts_data[:, 0]).mean().item()
            trend_direction = "upward" if trends > 0 else "downward"
            
            ts_mean = ts_data.mean(dim=1)
            ts_mean = ts_mean.unsqueeze(0).unsqueeze(0)
            autocorr = F.conv1d(ts_mean, ts_mean.flip(dims=[-1]), padding=ts_mean.shape[-1]-1)
            period = torch.argmax(autocorr[0, 0, autocorr.shape[-1]//2:]) + 1
            
            # Generate caption
            caption = (
                f"A time series visualization showing temporal patterns and characteristics. "
                f"The image displays value fluctuations through pixel intensity changes, "
                f"periodic patterns through repeating pixel structures, and stable segments "
                f"through consistent color regions. Dataset: {dataset}. "
                f"Key characteristics: sequence length={ts_data.shape[0]}, "
                f"variables={ts_data.shape[1]}, period={period.item()}, "
                f"trend={trend_direction}, value range=[{min_val:.2f}, {max_val:.2f}]."
            )
            
            # Extract features using CLIP
            inputs = processor(
                images=image_pil,
                text=caption,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                fused_feat = 0.7 * image_features + 0.3 * text_features
                fused_features.append(fused_feat)
        
        fused_features = torch.cat(fused_features, dim=0)
        ts_features_dict[dataset] = {'fused': fused_features}
        
        print(f"\nFeature Statistics for {dataset}:")
        print(f"Feature shape: {fused_features.shape}")
        print(f"Mean: {fused_features.mean().item():.4f}")
        print(f"Std: {fused_features.std().item():.4f}")
        print(f"Min: {fused_features.min().item():.4f}")
        print(f"Max: {fused_features.max().item():.4f}")
        
        print(f"Completed processing {dataset}")
    
    # Process COCO data
    coco_features = []
    coco_vision_features = []
    coco_text_features = []
    
    with open(caption_file, 'r') as f:
        captions = f.readlines()
    
    for i, line in enumerate(captions[:NUM_COCO_SAMPLES]):
        if i % 100 == 0:
            print(f"Processing COCO sample {i}")
            
        image_id = line.split(':')[0].split()[1]
        image_path = os.path.join(image_dir, f"image_{image_id}.jpg")
        if not os.path.exists(image_path):
            continue
            
        image = Image.open(image_path).convert('RGB')
        caption = line.split(':')[1].split('.')[0].strip()[:77]
        
        # Extract features using CLIP
        inputs = processor(
            images=image, 
            text=caption, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77
        )
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            fused_feat = 0.7 * image_features + 0.3 * text_features
            coco_features.append(fused_feat)
            
            # Extract vision features using ViT
            vit_inputs = vit_processor(images=image, return_tensors="pt")
            vit_outputs = vit_model(**vit_inputs)
            vision_emb = vit_outputs.last_hidden_state.mean(dim=1)
            vision_emb = vit_proj(vision_emb)
            coco_vision_features.append(vision_emb)
            
            # Extract text features using BERT
            bert_inputs = bert_tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
            bert_outputs = bert_model(**bert_inputs)
            text_emb = bert_outputs.last_hidden_state[:, 0, :]
            text_emb = bert_proj(text_emb)
            coco_text_features.append(text_emb)
    
    # Stack features
    coco_features = torch.cat(coco_features, dim=0)
    coco_vision_features = torch.cat(coco_vision_features, dim=0)
    coco_text_features = torch.cat(coco_text_features, dim=0)
    
    print("\nFeature Statistics for COCO:")
    print(f"CLIP Feature shape: {coco_features.shape}")
    print(f"Vision Feature shape: {coco_vision_features.shape}")
    print(f"Text Feature shape: {coco_text_features.shape}")
    
    # Calculate similarities
    print("\nPairwise Similarities between Datasets:")
    all_features = [coco_features] + [feat['fused'] for feat in ts_features_dict.values()]
    all_names = ['COCO'] + list(ts_features_dict.keys())
    
    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            feat1 = all_features[i]
            feat2 = all_features[j]
            feat1_norm = F.normalize(feat1, dim=-1)
            feat2_norm = F.normalize(feat2, dim=-1)
            similarity = torch.mean(torch.matmul(feat1_norm, feat2_norm.t()))
            print(f"{all_names[i]} - {all_names[j]} similarity: {similarity.item():.4f}")
    
    # Prepare features for visualization
    all_features = []
    feature_labels = []
    
    # Add time series features
    for dataset_name, features in ts_features_dict.items():
        features_np = features['fused'].detach().cpu().numpy()
        all_features.append(features_np)
        feature_labels.extend([dataset_name] * features_np.shape[0])
    
    # Add COCO features
    coco_np = coco_features.detach().cpu().numpy()
    coco_vision_np = coco_vision_features.detach().cpu().numpy()
    coco_text_np = coco_text_features.detach().cpu().numpy()
    
    all_features.extend([coco_np, coco_vision_np, coco_text_np])
    feature_labels.extend(['COCO-Pair'] * coco_np.shape[0])
    feature_labels.extend(['COCO-Image'] * coco_vision_np.shape[0])
    feature_labels.extend(['COCO-Text'] * coco_text_np.shape[0])
    
    # Visualize
    all_features = np.concatenate(all_features, axis=0)
    visualize_embeddings(all_features, feature_labels, save_path, 'UMAP Visualization of All Features')

if __name__ == "__main__":
    visualize_ts_embeddings_with_coco_samples() 