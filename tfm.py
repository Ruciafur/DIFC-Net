#v3
# tfm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def sinusoid_position_encoding(length: int, dim: int, device=None, dtype=None):
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ConvStem(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, mid_ratio: float = 0.25):
        super().__init__()
        mid1 = max(16, int(embed_dim * mid_ratio))
        mid2 = max(32, int(embed_dim * (mid_ratio * 2)))
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid1, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid1, mid2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):  # (B*T, C, H, W)
        return self.block(x)

class TrajectoryFeatureModel(nn.Module):
    """
    输入:  latent_seq [B, num_branches(=3), T, C(=4), H(=64), W(=64)]
    输出:  [B, hidden_dim]
    """
    def __init__(
        self,
        latent_channels: int = 4,
        latent_hw: int = 64,
        embed_dim: int = 192,     # 降低默认维度
        hidden_dim: int = 256,
        num_layers: int = 2,      # 降低层数
        num_heads: int = 4,       # 降低头数
        dropout: float = 0.1,
        num_branches: int = 3,
        use_checkpoint: bool = True,  # 开梯度检查点
        time_stride: int = 1,         # 时序下采样步长
        max_T: int | None = None,     # 时序截断
    ):
        super().__init__()
        self.num_branches = num_branches
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.time_stride = time_stride
        self.max_T = max_T

        self.conv_stem = ConvStem(latent_channels, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.token_dropout = nn.Dropout(dropout)

        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                               batch_first=True, dropout=dropout)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim * num_branches, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3)
        )

    def _transformer_ckpt(self, x):
        # 手动按层 checkpoint，省显存
        for layer in self.transformer.layers:
            x = checkpoint(layer, x)
        if getattr(self.transformer, "norm", None) is not None:
            x = self.transformer.norm(x)
        return x

    def _branch_forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        单分支: traj [B, T, C, H, W] -> [B, embed_dim]
        """
        # 时序下采样 & 截断
        if self.time_stride > 1:
            traj = traj[:, ::self.time_stride]
        if (self.max_T is not None) and (traj.size(1) > self.max_T):
            traj = traj[:, :self.max_T]

        B, T, C, H, W = traj.shape
        x = traj.reshape(B * T, C, H, W)
        x = self.conv_stem(x)                       # (B*T, D, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B*T, D)
        x = x.reshape(B, T, self.embed_dim)         # (B, T, D)

        pe = sinusoid_position_encoding(T, self.embed_dim, device=x.device, dtype=x.dtype)
        x = self.token_dropout(x + pe.unsqueeze(0))

        # Transformer + CKPT
        if self.use_checkpoint:
            x = self._transformer_ckpt(x)
        else:
            x = self.transformer(x)

        q = self.query_token.expand(B, -1, -1)      # (B,1,D)
        pooled, _ = self.attn_pool(q, x, x)         # (B,1,D)
        return pooled.squeeze(1)

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        B, num_branches, T, C, H, W = latent_seq.shape
        assert num_branches == self.num_branches
        feats = []
        for i in range(num_branches):
            feats.append(self._branch_forward(latent_seq[:, i]))  # [B, D]
        feats = torch.cat(feats, dim=1)  # [B, D*num_branches]
        return self.proj(feats)          # [B, hidden_dim]





# #v2
# # tfm.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TrajectoryFeatureModel(nn.Module):
#     """
#     输入: latent_seq [B, num_branches=3, T, C=4, H=64, W=64]
#     输出: [B, hidden_dim] (默认 256)

#     变更点:
#       1) 空间维度: 1x1 Conv 映射到 embed_dim, 全局平均池化得到 [B*T, embed_dim]
#       2) 时序维度: 使用 TransformerEncoder 编码
#       3) 池化方式: 用 Temporal Attention Pooling (可学习的 attention 权重, 替代 mean pool)
#       4) 正则化: Dropout + LayerNorm
#     """
#     def __init__(
#         self,
#         latent_channels: int = 4,
#         latent_hw: int = 64,
#         embed_dim: int = 512,
#         hidden_dim: int = 256,
#         num_layers: int = 4,
#         num_heads: int = 4,
#         num_branches: int = 3,
#         attn_dropout: float = 0.1,
#         proj_dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.num_branches = num_branches

#         # 1x1 conv 将 VAE latent (C=4) --> embed_dim，后面做全局平均池化
#         self.spatial_proj = nn.Conv2d(latent_channels, embed_dim, kernel_size=1)

#         # Transformer 编码 (时序)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dropout=attn_dropout,
#             batch_first=True,
#             norm_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # --- Temporal Attention Pooling ---
#         # 给每个时间步产生一个可学习的打分，再做 softmax 权重和
#         self.temporal_attn = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.GELU(),
#             nn.Dropout(attn_dropout),
#             nn.Linear(embed_dim // 2, 1)  # -> [B, T, 1]
#         )

#         # 分支拼接后线性降维到 hidden_dim
#         self.out_norm = nn.LayerNorm(embed_dim * num_branches)
#         self.out_dropout = nn.Dropout(proj_dropout)
#         self.proj = nn.Linear(embed_dim * num_branches, hidden_dim)

#     def forward(self, latent_seq):
#         """
#         latent_seq: [B, num_branches, T, C, H, W]
#         return: [B, hidden_dim]
#         """
#         B, num_branches, T, C, H, W = latent_seq.shape
#         assert num_branches == self.num_branches, f"Expect {self.num_branches} branches, got {num_branches}"

#         feats = []
#         for i in range(num_branches):
#             traj = latent_seq[:, i]                 # [B, T, C, H, W]
#             # 合并 batch 和时间做空间映射
#             traj = traj.reshape(B * T, C, H, W)     # [B*T, C, H, W]

#             # 保证 dtype 对齐
#             self.spatial_proj = self.spatial_proj.to(traj.device).to(traj.dtype)

#             x = self.spatial_proj(traj)             # [B*T, embed_dim, H, W]
#             x = x.mean(dim=(-1, -2))                # [B*T, embed_dim]
#             x = x.view(B, T, -1)                    # [B, T, embed_dim]

#             # 时序编码
#             x = self.transformer(x)                  # [B, T, embed_dim]

#             # --- Temporal Attention Pooling ---
#             # s: [B, T, 1] -> w: [B, T, 1]
#             scores = self.temporal_attn(x)
#             weights = torch.softmax(scores, dim=1)
#             pooled = (weights * x).sum(dim=1)        # [B, embed_dim]

#             feats.append(pooled)

#         # 跨分支拼接 -> 线性降维
#         feats = torch.cat(feats, dim=1)              # [B, embed_dim * num_branches]
#         feats = self.out_norm(feats)
#         feats = self.out_dropout(feats)
#         return self.proj(feats)                      # [B, hidden_dim]









# import torch
# import torch.nn as nn

# class TrajectoryFeatureModel(nn.Module):
#     def __init__(
#         self, 
#         latent_channels=4,    # 通道数按缓存格式设置
#         embed_dim=512, 
#         hidden_dim=256, 
#         num_layers=4, 
#         num_heads=4, 
#         num_branches=3
#     ):
#         super().__init__()
#         self.num_branches = num_branches
#         self.spatial_proj = nn.Conv2d(latent_channels, embed_dim, kernel_size=1)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.proj = nn.Linear(embed_dim * num_branches, hidden_dim)

#     def forward(self, latent_seq):
#         """
#         latent_seq: [B, num_branches, T, C, H, W]
#         C可为4（latent）或3（RGB），取决于缓存格式
#         """
#         B, num_branches, T, C, H, W = latent_seq.shape
#         feats = []
#         for i in range(num_branches):
#             traj = latent_seq[:, i]        # [B, T, C, H, W]
#             traj = traj.reshape(B * T, C, H, W)
#             # 确保卷积权重在同一设备与精度
#             self.spatial_proj = self.spatial_proj.to(traj.device).to(traj.dtype)
#             x = self.spatial_proj(traj)    # [B*T, embed_dim, h, w]
#             x = x.mean([-1, -2])           # [B*T, embed_dim]
#             x = x.reshape(B, T, -1)        # [B, T, embed_dim]
#             encoded = self.transformer(x)  # [B, T, embed_dim]
#             pooled = encoded.mean(dim=1)   # [B, embed_dim]
#             feats.append(pooled)
#         feats = torch.cat(feats, dim=1)    # [B, embed_dim * num_branches]
#         return self.proj(feats)            # [B, hidden_dim]












# import torch
# import torch.nn as nn

# class TrajectoryFeatureModel(nn.Module):
#     def __init__(self, latent_channels=4, latent_hw=64, embed_dim=512, hidden_dim=256, num_layers=4, num_heads=4, num_branches=3):
#         super().__init__()
#         self.num_branches = num_branches
#         self.spatial_proj = nn.Conv2d(latent_channels, embed_dim, kernel_size=1)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.proj = nn.Linear(embed_dim * num_branches, hidden_dim)

#     def forward(self, latent_seq):
#         """
#         latent_seq: [B, num_branches, T, C, H, W]
#         """
#         B, num_branches, T, C, H, W = latent_seq.shape
#         feats = []
#         for i in range(num_branches):
#             traj = latent_seq[:, i]          # [B, T, C, H, W]
#             b, t, c, h, w = traj.shape
#             traj = traj.reshape(B*T, c, h, w)  # 修正这里
#             self.spatial_proj = self.spatial_proj.to(traj.device).to(traj.dtype)
#             x = self.spatial_proj(traj)      # [B*T, embed_dim, h, w]
#             x = x.mean([-1, -2])             # [B*T, embed_dim]
#             x = x.reshape(B, T, -1)          # 修正这里
#             encoded = self.transformer(x)    # [B, T, embed_dim]
#             pooled = encoded.mean(dim=1)     # [B, embed_dim]
#             feats.append(pooled)
#         feats = torch.cat(feats, dim=1)      # [B, embed_dim * num_branches]
#         return self.proj(feats)              # [B, hidden_dim]