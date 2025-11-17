import torch.nn as nn

class ActiveCaptureFusion(nn.Module):
    def __init__(self, visual_dim, traj_dim, hidden_dim):
        super().__init__()
        self.proj_visual = nn.Linear(visual_dim, hidden_dim)
        self.proj_traj = nn.Linear(traj_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, visual_feat, traj_feat):
        visual_proj = self.proj_visual(visual_feat).unsqueeze(1)
        traj_proj = self.proj_traj(traj_feat).unsqueeze(1)
        fused_feat, _ = self.attn(visual_proj, traj_proj, traj_proj)
        return fused_feat.squeeze(1)