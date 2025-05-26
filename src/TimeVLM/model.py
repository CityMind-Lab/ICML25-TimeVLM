import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.TimeVLM.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.models_mae import *
from transformers.models.vilt import *

class PatchMemoryBank:
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        """
        Initialize the patch memory bank.
        
        Args:
            max_size (int): Maximum number of patches to store.
            patch_size (int): Size of each patch.
            feature_dim (int): Dimensionality of each patch feature.
            device (torch.device): Device to store memory bank on (CPU/GPU).
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)  # [100, d_model]
        self.ptr = 0

    def update(self, new_patches):
        """
        Update the patch memory bank with new patches.
        
        Args:
            new_patches (Tensor): New patches to add to the memory bank.
        """
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]
        
        if self.ptr + n > self.max_size:
            # Wrap around if the memory bank is full
            self.patches[self.ptr:] = new_patches_flat[:self.max_size - self.ptr]
            self.ptr = 0
        else:
            self.patches[self.ptr:self.ptr + n] = new_patches_flat
            self.ptr += n

    def retrieve(self, query_patches, top_k=5):
        """
        Retrieve the top-k most similar patches from the memory bank.
        
        Args:
            query_patches (Tensor): Query patches for retrieval.
            top_k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            retrieved_patches (Tensor): Retrieved patches from the memory bank.
            indices (Tensor): Indices of the retrieved patches.
        """
        query_flat = query_patches.mean(dim=1)  # [224, d_model]
        memory_flat = self.patches  # [100, d_model]
        
        similarity = torch.matmul(query_flat, memory_flat.T)  # [224, 100]
        _, indices = similarity.topk(top_k, dim=-1)
        
        retrieved_patches = self.patches[indices]
        return retrieved_patches, indices


class Model(nn.Module):
    """
    Multimodal Time Series Prediction Model based on CLIP.
    Processes image and text modalities for multimodal fusion and time series prediction.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config)
        self.device = torch.device('cuda:{}'.format(self.config.gpu))
        
        # Initialize patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=config.patch_memory_size,  # e.g., 100 patches
            patch_size=config.patch_len,
            feature_dim=config.d_model,
            device=self.device
        )
        
        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

    def _init_modules(self, config):
        self.patch_embedding = PatchEmbedding(
            config.d_model, 
            config.patch_len, 
            config.stride, 
            config.padding, 
            config.dropout
        )
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.multimodal_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Multimodal enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size * 2, config.d_model),  # Combine vision and text
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Cross-modal attention for feature enhancement
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Linear(config.pred_len, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
        self.layer_norm = nn.LayerNorm(config.d_model)


    def _compute_local_memory(self, patches):
        """Compute local memory by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank
        retrieved_patches, _ = self.patch_memory_bank.retrieve(patches, top_k=self.config.top_k)
        
        # Process retrieved patches with local MLP
        local_memory = self.local_memory_mlp(retrieved_patches)
        
        # Average over retrieved patches
        local_memory = local_memory.mean(dim=1, keepdim=True)
        
        # Residual connection with original patches
        local_memory = local_memory + patches
        
        return local_memory

    def _compute_global_memory(self, patches):
        """Compute global memory by aggregating information across all patches"""
        # Self-attention to capture global dependencies
        attn_output, _ = self.memory_attention(
            query=patches,
            key=patches,
            value=patches
        )
        
        # Temporal pooling to get global context
        global_memory = attn_output.mean(dim=1, keepdim=True)
        
        # Update patch memory bank with current patches
        self.patch_memory_bank.update(patches.detach())
        
        return global_memory

    def forward_prediction(self, x_enc, vision_embeddings, text_embeddings):
        B, L, n_vars = x_enc.shape
        
        # 1. Process temporal features
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))  # [B * n_vars, n_patches, d_model]
        
        # 2. Compute local and global memory
        local_memory = self._compute_local_memory(patches)  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(patches)  # [B * n_vars, 1, d_model]
        
        # 3. Combine local and global memory
        memory_features = local_memory + global_memory  # [B * n_vars, n_patches, d_model]

        # 4. Get temporal predictions
        memory_features = self.flatten(memory_features)  # [B * n_vars, head_nf]
        temporal_features = self.temporal_head(memory_features)  # [B, n_vars, d_model]
        memory_features = self.memory_head(memory_features)  # [B * n_vars, pred_len]
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, pred_len]
        
        # 5. Process multimodal features
        multimodal_features = torch.cat([vision_embeddings, text_embeddings], dim=-1)  # [B, hidden_size * 2]
        multimodal_features = self.multimodal_enhancement(multimodal_features)  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        
        # 6. Cross-modal attention enhancement
        temporal_features = temporal_features / torch.norm(temporal_features, dim=-1, keepdim=True)
        multimodal_features = multimodal_features / torch.norm(multimodal_features, dim=-1, keepdim=True)
        multimodal_features, _ = self.cross_attention(
            query=temporal_features,
            key=multimodal_features,
            value=multimodal_features
        )  # [B, n_vars, d_model]
        
        # 7. Normalize cross attention output
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        multimodal_features = self.multimodal_head(multimodal_features)  # [B, n_vars, pred_len]
        
        # 8. Compute gating weights
        combined_features = torch.cat([memory_features, multimodal_features], dim=-1)  # [B, n_vars, pred_len * 2]
        gate_weights = self.gate(combined_features)  # [B, n_vars, 2]
        
        # 9. Weighted fusion
        fused_features = (
            gate_weights[:, :, 0:1] * memory_features +
            gate_weights[:, :, 1:2] * multimodal_features
        ) # [B, n_vars, pred_len]
        
        # 10. Final fusion
        predictions = self.fusion_layer(
            torch.cat([memory_features, fused_features], dim=-1)
        ) + memory_features  # [B, n_vars, pred_len]
        
        return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        
        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)
        
        # Convert time series data to images and generate text prompts
        images = self.vision_augmented_learner(x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity)
        prompts = self.text_augmented_learner(x_enc, self.config.content, self.config.pred_len, self.config.seq_len)
        
        # Process inputs with the VLM
        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(B, images, prompts)
        
        # Main prediction branch
        predictions = self.forward_prediction(x_enc, vision_embeddings, text_embeddings)
        
        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y

    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    def text_augmented_learner(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.
        Each variable in the time series will have its own prompt.
        """
        B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

        # Initialize a list to store prompts for each batch
        prompts = []
    
        # Calculate overall statistics for each batch
        for b in range(B):
            # Calculate statistics for the current batch
            min_value = torch.min(x_enc[b]).item()  # Overall minimum value for the batch
            max_value = torch.max(x_enc[b]).item()  # Overall maximum value for the batch
            median_value = torch.median(x_enc[b]).item()  # Overall median value for the batch
            trend = x_enc[b].diff(dim=0).sum().item()  # Overall trend for the batch

            # Determine the overall trend direction
            trend_direction = "upward" if trend > 0 else "downward"
                
            prompt_parts = [
                "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_manager.max_input_text_length] if len(prompt) > self.vlm_manager.max_input_text_length else prompt
            prompts.append(prompt)  

        return prompts

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.config.save_images:
            self.save_images(images)

        return images
    
    @staticmethod
    def _normalize_images(images):
        """
        Normalize image tensors to [0, 255] as uint8.
        Assumes images are in [0, 1] or need to be scaled.
        
        Args:
        - images (Tensor): Input images with shape [B, C, H, W]
        
        Returns:
        - Tensor: Normalized images as uint8 with shape [B, C, H, W]
        """
        # Compute min and max per image across all channels and spatial dimensions
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        
        return images

    @torch.no_grad()
    def save_images(self, images):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved.
        - batch_idx: Index of the current batch.
        """
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to [H, W, C] and scale to [0, 255]
            img_tensor = img_tensor.astype(np.uint8)
            img = Image.fromarray(img_tensor)
            img.save(os.path.join(save_dir, f"image_{i}.png"))


def check_image_channel(np_img):
    """
    Check the number of channels in an image and adjust the dimension order.

    Args:
    - np_img: Input image.

    Returns:
    - np_img: Adjusted image.
    - mode: Image mode (e.g., 'RGB', 'L').
    """
    # Check channel count and adjust dimension order
    if np_img.shape[0] == 3:
        # RGB image: Convert from [C, H, W] to [H, W, C]
        np_img = np.transpose(np_img, (1, 2, 0))  # [224, 224, 3]
        mode = 'RGB'
    elif np_img.shape[0] == 1:
        # Grayscale image: Convert from [C, H, W] to [H, W]
        np_img = np.squeeze(np_img, 0)  # [224, 224]
        mode = 'L'
    else:
        print(f"Unexpected number of channels: {np_img.shape[0]} for image")

    return np_img, mode
