import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=4, spatial_size=16, add_class_token=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, batch_first=True
        )
        self.spatial_size = spatial_size
        self.add_class_token = add_class_token
        if add_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img_feat, recon_feat, noise_feat, return_attn=False):
        img_feat   = F.adaptive_avg_pool2d(img_feat, (self.spatial_size, self.spatial_size))
        recon_feat = F.adaptive_avg_pool2d(recon_feat, (self.spatial_size, self.spatial_size))
        noise_feat = F.adaptive_avg_pool2d(noise_feat, (self.spatial_size, self.spatial_size))
        B, C, H, W = img_feat.shape
        N = H * W

        img_vec   = img_feat.flatten(2).transpose(1, 2)
        recon_vec = recon_feat.flatten(2).transpose(1, 2)
        noise_vec = noise_feat.flatten(2).transpose(1, 2)
        feats = torch.cat([img_vec, recon_vec, noise_vec], dim=1)  # [B, 3N, C]

        if self.add_class_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            feats = torch.cat([cls_token, feats], dim=1)

        attn_out, attn_weights = self.attn(
            feats, feats, feats, need_weights=True, average_attn_weights=False
        )  # attn_weights: [B, num_heads, Lq, Lk]

        fused = attn_out[:, 0] if self.add_class_token else attn_out.mean(dim=1)

        if not return_attn:
            return fused

        # 平均所有head -> [B, Lq, Lk]
        attn_map = attn_weights.mean(dim=1)
        return fused, attn_map





#训练使用
# class CrossModalAttention(nn.Module):
#     def __init__(self, feat_dim, num_heads=4, spatial_size=16, add_class_token=False):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
#         self.spatial_size = spatial_size
#         self.add_class_token = add_class_token
#         if add_class_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
#             nn.init.trunc_normal_(self.cls_token, std=0.02)

#     def forward(self, img_feat, recon_feat, noise_feat):
#         img_feat   = F.adaptive_avg_pool2d(img_feat, (self.spatial_size, self.spatial_size))
#         recon_feat = F.adaptive_avg_pool2d(recon_feat, (self.spatial_size, self.spatial_size))
#         noise_feat = F.adaptive_avg_pool2d(noise_feat, (self.spatial_size, self.spatial_size))
#         B, C, H, W = img_feat.shape
#         img_vec   = img_feat.flatten(2).transpose(1, 2)
#         recon_vec = recon_feat.flatten(2).transpose(1, 2)
#         noise_vec = noise_feat.flatten(2).transpose(1, 2)
#         feats = torch.cat([img_vec, recon_vec, noise_vec], dim=1)  # [B, 3*HW, C]

#         if self.add_class_token:
#             cls_token = self.cls_token.expand(B, -1, -1)           # [B, 1, C]
#             feats = torch.cat([cls_token, feats], dim=1)           # [B, 1+3*HW, C]

#         attn_out, _ = self.attn(feats, feats, feats)
#         if self.add_class_token:
#             fused = attn_out[:, 0]     # 取 class token 输出
#         else:
#             fused = attn_out.mean(dim=1)
#         return fused  # [B, C]


