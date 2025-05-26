import torch
import torch.nn as nn

class TemporalProjection(nn.Module):
    def __init__(self, fusion_dim, d_model, pred_len, nhead=8, dropout=0.1):
        super().__init__()
        # Projection layer to map fusion_dim to d_model
        self.projection = nn.Linear(fusion_dim, d_model)
        
        # Temporal Feature Extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, pred_len, kernel_size=3, padding=1)
        )
        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):  # x: [B, fusion_dim]
        # Project x to [B, d_model]
        x = self.projection(x)  # [B, fusion_dim] => [B, d_model]

        # Reshape x to [B, 1, d_model] for Conv1d
        x = x.unsqueeze(1)  # [B, d_model] => [B, 1, d_model]

        # Temporal Feature Extraction
        conv_out = self.conv_block(x)  # [B, 1, d_model] => [B, pred_len, d_model]

        # Permute for MultiheadAttention: [B, pred_len, d_model] => [pred_len, B, d_model]
        conv_out = conv_out.permute(1, 0, 2)

        # Self-Attention Layer
        attn_in = self.norm(conv_out)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in)  # [pred_len, B, d_model]

        # Permute back: [pred_len, B, d_model] => [B, pred_len, d_model]
        attn_out = attn_out.permute(1, 0, 2)

        # LayerNorm
        out = self.norm(attn_out)
        return out
    

class MultiscaleTemporalProjection(nn.Module):
    def __init__(self, fusion_dim, d_model, pred_len, nhead=8, dropout=0.1):
        super().__init__()
        
        # Enhanced multi-scale temporal feature extraction
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(fusion_dim, fusion_dim, kernel_size=3, padding=1, dilation=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(fusion_dim)
            ),
            nn.Sequential(
                nn.Conv1d(fusion_dim, fusion_dim, kernel_size=5, padding=4, dilation=2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(fusion_dim)
            ),
            nn.Sequential(
                nn.Conv1d(fusion_dim, fusion_dim, kernel_size=7, padding=12, dilation=4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(fusion_dim)
            )
        ])
        
        # Frequency domain processing branch
        self.freq_branch = nn.Sequential(
            nn.Conv1d(fusion_dim*3, fusion_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_dim)
        )
        
        # Feature compression and fusion
        self.fusion_compress = nn.Sequential(
            nn.Conv1d(fusion_dim*3, fusion_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_dim)
        )
        
        # Adaptive feature fusion with attention
        self.fusion_attention = nn.MultiheadAttention(fusion_dim, nhead, dropout=dropout)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        # Enhanced temporal attention with residual connections
        self.temporal_attn = nn.MultiheadAttention(fusion_dim, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.norm3 = nn.LayerNorm(fusion_dim)
            
        # Final projection with skip connections
        self.final_proj = nn.Sequential(
            nn.Conv1d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(fusion_dim),
            nn.Conv1d(fusion_dim, pred_len, kernel_size=3, padding=1),
            nn.GELU()
        )

        
    def feature_fusion(self, x):
        # Reshape for attention
        x = x.permute(2, 0, 1)  # [seq_len, batch, features]
        # Apply attention
        attn_out, _ = self.fusion_attention(x, x, x)
        attn_out = attn_out + x  # Residual connection
        attn_out = self.fusion_norm(attn_out)
        # Reshape back
        attn_out = attn_out.permute(1, 2, 0)  # [batch, features, seq_len]
        return attn_out
        
    def forward(self, x):
        # Multi-scale feature extraction
        conv_features = []
        for conv_block in self.conv_blocks:
            conv_features.append(conv_block(x))
        
        # Feature fusion
        fused_features = torch.cat(conv_features, dim=1)        
        fused_features = self.fusion_compress(fused_features)        
        fused_features = self.feature_fusion(fused_features)
                        
        # Temporal attention
        fused_features = fused_features.permute(0, 2, 1)
        attn_in = self.norm1(fused_features)
        attn_out, _ = self.temporal_attn(attn_in, attn_in, attn_in)
        attn_out = attn_out + fused_features  # Residual connection
        attn_out = self.norm2(attn_out)
        attn_out = attn_out.permute(0, 2, 1)
        
        # Final projection
        out = self.final_proj(attn_out)
        return out
