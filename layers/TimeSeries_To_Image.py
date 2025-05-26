import einops
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # Resize the time series data
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Convert to 3-channel image
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images


def time_series_to_image_with_fft_and_wavelet(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors using FFT and Wavelet transforms.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    def _apply_fourier_transform(x_2d):
        """
        Apply Fourier transform to the input 2D time series data.
        """
        x_fft = torch.fft.fft(x_2d, dim=-1)
        x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
        return x_fft_abs

    def _apply_wavelet_transform(x_2d):
        """
        Apply wavelet transform to the input 2D time series data.
        """
        dwt = DWTForward(J=1, wave='haar')
        # cA: Low-frequency components, cD: High-frequency components
        cA, cD = dwt(x_2d)  # [B * nvars, 1, f, p]
        cD_reshaped = cD[0].squeeze(1)  # [B * nvars, 3, f, p]
        # Concatenate low-frequency and high-frequency components
        wavelet_result = torch.cat([cA, cD_reshaped], dim=1)  # [B * nvars, 4, f, p]
        # Average across the channel dimension to reduce to 1 channel
        wavelet_result = wavelet_result.mean(dim=1, keepdim=True)  # [B * nvars, 1, f, p]
        return wavelet_result
    
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # Resize the time series data
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Apply Fourier transform or wavelet transform
    x_fft = _apply_fourier_transform(x_2d)
    x_wavelet = _apply_wavelet_transform(x_2d)
    # Resize the Fourier or wavelet transformed data as image input using interpolation
    x_resized_fft = F.interpolate(x_fft, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x_resized_wavelet = F.interpolate(x_wavelet, size=(image_size, image_size), mode='bilinear', align_corners=False)
    # Concatenate along the channel dimension to form a 3-channel image
    images = torch.concat([x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1)  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images