import torch
import torch.nn as nn

class EnhancedRACModule(nn.Module):
    def __init__(self, in_dim=256, reduction=8):
        super().__init__()
        hidden = max(1, in_dim // reduction)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, in_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.reduce = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, img_feat, recon_feat, noise_feat):
        residual_ir = torch.abs(img_feat - recon_feat)
        residual_in = torch.abs(img_feat - noise_feat)
        residual_rn = torch.abs(recon_feat - noise_feat)
        residual = residual_ir + residual_in + residual_rn
        residual = self.reduce(residual)
        chn_attn = self.channel_attn(residual)
        max_pool, _ = torch.max(residual, dim=1, keepdim=True)
        avg_pool = torch.mean(residual, dim=1, keepdim=True)
        spa_attn = self.spatial_attn(torch.cat([max_pool, avg_pool], dim=1))
        attn = chn_attn * spa_attn
        return img_feat + img_feat * attn  # [B, C, H, W]