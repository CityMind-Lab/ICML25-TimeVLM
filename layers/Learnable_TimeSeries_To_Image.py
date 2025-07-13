from sqlite3 import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import numpy as np

class TimeSeriesVisualizer:
    """Visualization tools for time series to image conversion"""
    
    @staticmethod
    def plot_feature_maps(features, title="Feature Maps"):
        """Plot feature maps from intermediate layers"""
        if not isinstance(features, torch.Tensor):
            return
            
        # Convert to numpy and normalize
        features = features.detach().cpu().numpy()
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # Plot first batch item
        num_channels = min(4, features.shape[1])
        if num_channels == 1:
            plt.figure(figsize=(5, 5))
            plt.imshow(features[0, 0], cmap='viridis')
            plt.axis('on')
            plt.title('Channel 1')
        else:
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(features[0, i], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {i+1}')
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        import os
        os.makedirs('ts-images/ts-visualizer', exist_ok=True)
        filename = f"ts-images/ts-visualizer/{title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        plt.close()

    @staticmethod
    def plot_attention(attention_map, title="Attention Map"):
        """Plot attention weights"""
        if not isinstance(attention_map, torch.Tensor):
            return
            
        attention_map = attention_map.detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map[0, 0], cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()

class LearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]
        
        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodic
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D Convolution processing
        x_enc = x_enc.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_enc = F.relu(self.conv2d_1(x_enc))
        x_enc = F.relu(self.conv2d_2(x_enc))
        
        # Resize to target image size
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x_enc  # [B, output_channels, H, W]


class MultiChannalLearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(MultiChannalLearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]
        
        # Combine raw data with periodic encoding
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, periodicity_encoding], dim=-1)  # [B, L, D, 3]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 3, L]
        x_enc = x_enc.reshape(B * D, 3, L)  # [B*D, 3, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        
        # Add channel dimension for 2D processing
        x_enc = x_enc.unsqueeze(2)  # [B*D, hidden_dim, 1, L]
        
        # 2D Convolution processing
        x_enc = F.relu(self.conv2d_1(x_enc))
        x_enc = F.relu(self.conv2d_2(x_enc))
        
        # Resize to target image size
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x_enc  # [B*D, output_channels, H, W]

class MultiscaleLearnableTimeSeriesToImage(nn.Module):
    """Enhanced learnable module for time series to image conversion with multi-scale features"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(MultiscaleLearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # Multi-scale 1D convolutions with different dilations
        self.conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU()
            ),
        ])
        
        # Frequency domain convolution
        self.fft_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Residual connection
        self.residual = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # 2D convolution blocks
        self.conv2d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=3, padding=1),
                nn.ReLU()
            ),
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//8, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv2d_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Discrete wavelet transform
        self.dwt = DWTForward(J=1, wave='haar')
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(18*hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # Projection layers
        self.proj_for_attention = nn.Conv2d(3, hidden_dim, kernel_size=1)
        self.final_proj_to_output = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]

        # Combine raw data with periodic encoding
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, periodicity_encoding], dim=-1)  # [B, L, D, 3]

        # Reshape for multi-scale processing
        x_enc = x_enc.view(B * D, 3, L)

        # Multi-scale 1D convolution features
        conv_features = []
        for conv_block in self.conv1d_blocks:
            conv_features.append(conv_block(x_enc))
        x_enc = torch.cat(conv_features, dim=1)  # Concatenate along channel dim
        
        # Frequency domain processing
        freq_features = self.freq_processing(x_enc)
        
        # Downsample and fuse features
        x_enc_downsampled = x_enc[:, :, ::2]  # Match freq feature length
        x_enc = torch.cat([x_enc_downsampled, freq_features], dim=1)
        x_enc = self.feature_fusion(x_enc)

        # Prepare for 2D processing
        x_enc = x_enc.unsqueeze(2)  # Add spatial dimension

        # 2D convolution processing
        for conv2d_block in self.conv2d_blocks:
            x_enc = conv2d_block(x_enc)
        
        # Attention mechanism
        x_enc = self.proj_for_attention(x_enc)
        attention_map = self.attention(x_enc)
        x_enc = x_enc * attention_map  # Apply attention weights

        # Final projection and resizing
        x_enc = self.final_proj(x_enc)
        x_enc = self.final_proj_to_output(x_enc)
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Reshape and average over variables
        x_enc = x_enc.view(B, D, self.output_channels, self.image_size, self.image_size)
        x_enc = x_enc.mean(dim=1)  # [B, output_channels, H, W]
                
        return x_enc

    def freq_processing(self, x):
        """Process frequency domain features using FFT and wavelet transforms"""
        x = x.float()
        
        # FFT magnitude features
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = torch.abs(x_fft)
        x_fft = x_fft[..., :x.shape[2] // 2]  # Match wavelet feature length
        
        # Wavelet transform features
        x_2d = x.unsqueeze(2)
        cA, cD = self.dwt(x_2d)
        cD_reshaped = cD[0].squeeze(3)  # [B*D, C, 3, L//2]
        wavelet_features = torch.cat([cA, cD_reshaped], dim=2)  # [B*D, C, 4, L//2]
        
        # Match wavelet dimensions
        x_fft = x_fft.unsqueeze(2)  # [B*D, C, 1, L//2]
        x_fft = x_fft.expand(-1, -1, 4, -1)  # [B*D, C, 4, L//2]
        
        # Combine frequency features
        freq_features = torch.cat([x_fft, wavelet_features], dim=1)
        freq_features = freq_features.view(freq_features.shape[0], -1, freq_features.shape[-1])
        
        return freq_features


    def final_proj(self, x):
        """Final projection with residual connection"""
        identity = x
        x = self.conv2d_2(x)
        return x + identity  # Residual connection
