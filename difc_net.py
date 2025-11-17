#v3
# difc_net.py
import torch
import torch.nn as nn

from tfm import TrajectoryFeatureModel          # 已支持 use_checkpoint / time_stride / max_T 等
from rac import EnhancedRACModule
from cross_modal_attention import CrossModalAttention
from acf import ActiveCaptureFusion
from classifier import Classifier


class DIFCNet(nn.Module):
    """
    DIFC-Net 主模型（离线缓存轨迹版本）
    输入：
        - img_traj   : [B, T, 4, 64, 64]
        - recon_traj : [B, T, 4, 64, 64]
        - noise_traj : [B, T, 4, 64, 64]
    输出：
        - logits     : [B, 1]
    关键特性：
        1) TFM 为可配置（embed_dim / num_layers / heads / dropout / use_checkpoint / time_stride / max_T）
        2) 帧池化策略可配置：first / mean / avgk
        3) CrossModalAttention 可配置 head 数、空间池化大小
    """

    def __init__(
        self,
        proj_dim: int = 256,
        # 传给 TrajectoryFeatureModel 的配置（建议用我的降显存默认）
        tfm_kwargs: dict | None = None,
        # 帧池化策略
        frame_pool: str = "first",   # ["first", "mean", "avgk"]
        frame_k: int = 4,            # avgk 时使用，计算前 K 帧的均值
        # Cross-Modal attention 参数（与 cross_modal_attention.py 对应）
        cross_heads: int = 4,
        cross_spatial: int = 16,
    ):
        super().__init__()

        # -------- 1) 构造 TFM（带梯度检查点与时序下采样能力） --------
        tfm_defaults = dict(
            embed_dim=192, hidden_dim=256,
            num_layers=2, num_heads=4,
            dropout=0.1,
            num_branches=3,
            use_checkpoint=True,   # 大幅节省显存
            time_stride=1,         # 若显存紧张可设为 2
            max_T=None,            # 或者直接截断到某个 T
        )
        if tfm_kwargs is not None:
            tfm_defaults.update(tfm_kwargs)
        self.tfm = TrajectoryFeatureModel(**tfm_defaults)

        self.frame_pool = frame_pool
        self.frame_k = max(1, int(frame_k))
        self.traj_dim = tfm_defaults["hidden_dim"]  # 下游 ACF 的 traj_dim 需与此一致

        # -------- 2) 统一特征投影（注意：in_channels=4） --------
        self.feat_proj = nn.Conv2d(4, proj_dim, kernel_size=3, stride=1, padding=1)
        # 轻量归一化提升稳定性（可注释掉）
        self.feat_gn = nn.GroupNorm(num_groups=min(32, proj_dim), num_channels=proj_dim)

        # -------- 3) 局部差异增强（RAC），与跨模态注意力 --------
        self.rac = EnhancedRACModule(in_dim=proj_dim)
        self.cross_attn = CrossModalAttention(
            feat_dim=proj_dim, num_heads=cross_heads, spatial_size=cross_spatial
        )

        # -------- 4) 融合与分类 --------
        self.acf = ActiveCaptureFusion(
            visual_dim=proj_dim, traj_dim=self.traj_dim, hidden_dim=256
        )
        self.classifier = Classifier(input_dim=256)

    # -------------------- 内部工具：帧池化 --------------------
    def _pool_frames(self, traj: torch.Tensor) -> torch.Tensor:
        """
        traj: [B, T, 4, H, W] -> [B, 4, H, W]
        支持:
            - "first": 取第 0 帧
            - "mean" : 所有帧平均
            - "avgk" : 前 K 帧平均
        """
        if self.frame_pool == "first":
            return traj[:, 0]                                     # [B, 4, H, W]
        elif self.frame_pool == "mean":
            return traj.mean(dim=1)                               # [B, 4, H, W]
        elif self.frame_pool == "avgk":
            k = min(self.frame_k, traj.size(1))
            return traj[:, :k].mean(dim=1)                        # [B, 4, H, W]
        else:
            # 兜底：使用 first
            return traj[:, 0]

    # -------------------- 前向 --------------------
    def forward(self, img_traj, recon_traj, noise_traj):
        """
        img_traj / recon_traj / noise_traj : [B, T, 4, 64, 64]
        """
        # ---- 视觉分支：从轨迹中选帧（或池化多帧），再投影 -> RAC -> CrossAttn ----
        img0   = self._pool_frames(img_traj)    # [B, 4, H, W]
        recon0 = self._pool_frames(recon_traj)
        noise0 = self._pool_frames(noise_traj)

        img_proj   = self.feat_gn(self.feat_proj(img0))
        recon_proj = self.feat_gn(self.feat_proj(recon0))
        noise_proj = self.feat_gn(self.feat_proj(noise0))

        # 局部差异注意力（输出仍是 [B, C, H, W]）
        _ = self.rac(img_proj, recon_proj, noise_proj)
        # 跨模态注意力（输出 [B, C] 的全局特征）
        cross_fused = self.cross_attn(img_proj, recon_proj, noise_proj)  # [B, proj_dim]

        # ---- 轨迹分支：三路堆叠 -> [B, 3, T, 4, 64, 64] -> TFM -> [B, traj_dim] ----
        traj_6d = torch.stack([img_traj, recon_traj, noise_traj], dim=1).contiguous()
        traj_feat = self.tfm(traj_6d)  # [B, traj_dim]

        # ---- 融合 + 分类 ----
        fused_feat = self.acf(cross_fused, traj_feat)  # [B, 256]
        logits = self.classifier(fused_feat)           # [B, 1]
        return logits





#v2
# import torch
# import torch.nn as nn
# from tfm import TrajectoryFeatureModel
# from rac import EnhancedRACModule
# from cross_modal_attention import CrossModalAttention
# from acf import ActiveCaptureFusion
# from classifier import Classifier

# class DIFCNet(nn.Module):
#     def __init__(self, proj_dim=256):
#         super().__init__()
#         self.tfm = TrajectoryFeatureModel()
#         self.feat_proj = nn.Conv2d(4, proj_dim, kernel_size=3, stride=1, padding=1)  # in_channels=4!
#         self.rac = EnhancedRACModule(in_dim=proj_dim)
#         self.cross_attn = CrossModalAttention(feat_dim=proj_dim, num_heads=4)
#         self.acf = ActiveCaptureFusion(visual_dim=proj_dim, traj_dim=256, hidden_dim=256)
#         self.classifier = Classifier(input_dim=256)

#     def forward(self, img_traj, recon_traj, noise_traj):  # [B, T, 4, 64, 64]
#         img0   = img_traj[:, 0]   # [B, 4, 64, 64]
#         recon0 = recon_traj[:, 0]
#         noise0 = noise_traj[:, 0]

#         img_proj = self.feat_proj(img0)
#         recon_proj = self.feat_proj(recon0)
#         noise_proj = self.feat_proj(noise0)

#         rac_feat = self.rac(img_proj, recon_proj, noise_proj)           # [B, proj_dim, H, W]
#         cross_fused = self.cross_attn(img_proj, recon_proj, noise_proj) # [B, proj_dim]
#         traj_feat = self.tfm(torch.stack([img_traj, recon_traj, noise_traj], dim=1))  # [B, 3, T, 4, 64, 64]

#         fused_feat = self.acf(cross_fused, traj_feat)
#         logits = self.classifier(fused_feat)
#         return logits



# class DIFCNet(nn.Module):
#     def __init__(self, proj_dim=256):
#         super().__init__()
#         self.tfm = TrajectoryFeatureModel()
#         self.feat_proj = nn.Conv2d(3, proj_dim, kernel_size=3, stride=1, padding=1)
#         self.rac = EnhancedRACModule(in_dim=proj_dim)
#         self.cross_attn = CrossModalAttention(feat_dim=proj_dim, num_heads=4)
#         self.acf = ActiveCaptureFusion(visual_dim=proj_dim, traj_dim=256, hidden_dim=256)
#         self.classifier = Classifier(input_dim=256)

#     def forward(self, img_traj, recon_traj, noise_traj):
#         # img_traj, recon_traj, noise_traj: [B, T, C, H, W]
#         # 取第0帧（如有需求可改成平均池化或其他策略）
#         img0   = img_traj[:, 0]     # [B, C, H, W]
#         recon0 = recon_traj[:, 0]
#         noise0 = noise_traj[:, 0]

#         img_proj   = self.feat_proj(img0)      # [B, proj_dim, H, W]
#         recon_proj = self.feat_proj(recon0)
#         noise_proj = self.feat_proj(noise0)

#         rac_feat = self.rac(img_proj, recon_proj, noise_proj)               # [B, proj_dim, H, W]
#         cross_fused = self.cross_attn(img_proj, recon_proj, noise_proj)     # [B, proj_dim]
#         # 轨迹整体融合
#         traj = torch.stack([img_traj, recon_traj, noise_traj], dim=1)       # [B, 3, T, C, H, W]
#         traj_feat = self.tfm(traj)                                          # [B, 256]

#         fused_feat = self.acf(cross_fused, traj_feat)
#         logits = self.classifier(fused_feat)
#         return logits












# import torch.nn as nn
# from ltt import LatentTrajectoryTracker
# from tfm import TrajectoryFeatureModel
# from rac import EnhancedRACModule
# from cross_modal_attention import CrossModalAttention
# from acf import ActiveCaptureFusion
# from classifier import Classifier

# class DIFCNet(nn.Module):
#     def __init__(self, proj_dim=256):
#         super().__init__()
#         self.ltt = LatentTrajectoryTracker()
#         self.tfm = TrajectoryFeatureModel()
#         # 新增：统一特征投影层
#         self.feat_proj = nn.Conv2d(3, proj_dim, kernel_size=3, stride=1, padding=1)
#         # 下游所有视觉处理通道数都为 proj_dim
#         self.rac = EnhancedRACModule(in_dim=proj_dim)
#         self.cross_attn = CrossModalAttention(feat_dim=proj_dim, num_heads=4)
#         self.acf = ActiveCaptureFusion(visual_dim=proj_dim, traj_dim=256, hidden_dim=256)
#         self.classifier = Classifier(input_dim=256)

#     def forward(self, img, recon_img, noise_img, prompt):
#         # 轨迹特征
#         latent_seq = self.ltt.track(img, recon_img, noise_img, prompt)
#         traj_feat = self.tfm(latent_seq)
#         # 前置特征投影
#         img_proj = self.feat_proj(img)
#         recon_proj = self.feat_proj(recon_img)
#         noise_proj = self.feat_proj(noise_img)
#         # RAC高维特征
#         rac_feat = self.rac(img_proj, recon_proj, noise_proj)  # [B, proj_dim, H, W]
#         # CrossModalAttention高维特征
#         cross_fused = self.cross_attn(img_proj, recon_proj, noise_proj)  # [B, proj_dim]
#         # 融合视觉（cross_fused）和轨迹特征
#         fused_feat = self.acf(cross_fused, traj_feat)
#         logits = self.classifier(fused_feat)
#         return logits