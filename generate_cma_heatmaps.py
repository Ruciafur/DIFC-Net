import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from difc_net import DIFCNet
from cross_modal_attention import CrossModalAttention

# ===============================
# Config
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

PROJ_DIM      = 256          # 与 DIFCNet(proj_dim) 一致
SPATIAL_SIZE  = 16           # 与 CMA 中 spatial_size 一致
INPUT_SIZE    = 256          # 输入可视大小（建议能被16整除）
ADD_CLS_TOKEN = True         # 与 CMA 配置一致

IMG_DIR   = "cma_sample/inputs/img"
RECON_DIR = "cma_sample/inputs/recon"
NOISE_DIR = "cma_sample/inputs/noise"
SAVE_DIR  = "cma_sample/outputs/vis4"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# Utils
# ===============================
class ToFourChannels(nn.Module):
    """将 RGB 扩展成 4 通道（RGB + Gray），以适配 DIFCNet 的 in_channels=4。"""
    def __init__(self, mode="gray"):
        super().__init__()
        self.mode = mode
    def forward(self, x):  # x: [B,3,H,W]
        if x.size(1) == 4:
            return x
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        if self.mode == "gray":
            y = 0.2989*r + 0.5870*g + 0.1140*b
        elif self.mode == "zeros":
            y = torch.zeros_like(r)
        else:
            y = torch.ones_like(r)
        return torch.cat([x, y], dim=1)

def load_image(path, size=INPUT_SIZE):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    ten = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return ten.unsqueeze(0).to(device), img  # ([1,3,H,W], uint8 RGB)

def norm_np(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)

def overlay_heatmap(orig_rgb_uint8, attn01, title):
    h, w = orig_rgb_uint8.shape[:2]
    attn01 = cv2.resize(attn01, (w, h))
    heat = cv2.applyColorMap((attn01*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = (0.5*orig_rgb_uint8 + 0.5*heat).astype(np.uint8)
    return overlay

def save_four_panel(img_np, cma_img, cma_recon, cma_noise, gradcam, save_path, suptitle=None):
    plt.figure(figsize=(10, 10))
    panels = [
        ("CMA Image",  cma_img),
        ("CMA Recon",  cma_recon),
        ("CMA Noise",  cma_noise),
        ("Grad-CAM (Fused decision)", gradcam),
    ]
    for i, (t, vis) in enumerate(panels, 1):
        plt.subplot(2, 2, i)
        plt.imshow(vis); plt.title(t); plt.axis("off")
    if suptitle: plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()

# ===============================
# Models
# ===============================
difc = DIFCNet(proj_dim=PROJ_DIM).to(device).eval()
to4  = ToFourChannels().to(device).eval()
cma  = CrossModalAttention(
    feat_dim=PROJ_DIM, num_heads=4, spatial_size=SPATIAL_SIZE, add_class_token=ADD_CLS_TOKEN
).to(device).eval()

# ===============================
# Hook for Grad-CAM (对融合后的最终输出回溯)
# 我们选择在 feat_gn 的输出上做 Grad-CAM（空间层，紧邻 VCP 起始）
# 该层在 forward 中会被调用三次（img/recon/noise），我们取第一次（img 分支）
# ===============================
acts, grads = [], []
call_counter = {"feat_gn": 0}

def fwd_hook(module, inp, out):
    # 记录第1次调用（对应 img 分支）
    if call_counter["feat_gn"] == 0:
        acts.append(out)  # [B,C,H,W]
    call_counter["feat_gn"] += 1

def bwd_hook(module, gin, gout):
    # 只记录与 acts 对应的梯度（第1次）
    if len(grads) == 0:
        grads.append(gout[0])  # [B,C,H,W]

handle_f = difc.feat_gn.register_forward_hook(fwd_hook)
handle_b = difc.feat_gn.register_full_backward_hook(bwd_hook)

# ===============================
# CMA 三路注意力 + GradCAM 生成流程
# ===============================
img_names = sorted(os.listdir(IMG_DIR))
print(f"Found {len(img_names)} samples")

for fname in tqdm(img_names, desc="CMA & Grad-CAM"):
    name = os.path.splitext(fname)[0]
    ip, rp, np_ = (os.path.join(IMG_DIR, fname),
                   os.path.join(RECON_DIR, fname),
                   os.path.join(NOISE_DIR, fname))
    if not (os.path.exists(rp) and os.path.exists(np_)):
        print(f"Skip {fname}: missing recon/noise.")
        continue

    # 1) 读取图像
    img_t,   img_np   = load_image(ip)
    recon_t, recon_np = load_image(rp)
    noise_t, noise_np = load_image(np_)

    # 2) 4通道适配 + VCP早期特征（与 DIFCNet 保持一致）
    with torch.no_grad():
        img4   = to4(img_t)
        recon4 = to4(recon_t)
        noise4 = to4(noise_t)

        img_proj   = difc.feat_gn(difc.feat_proj(img4))
        recon_proj = difc.feat_gn(difc.feat_proj(recon4))
        noise_proj = difc.feat_gn(difc.feat_proj(noise4))

        # RAC 增强（与训练路径一致）
        _ = difc.rac(img_proj, recon_proj, noise_proj)

        # 3) CMA：取 CLS→keys 的注意力，拆分为三路并可视化
        _, attn_map = cma(img_proj, recon_proj, noise_proj, return_attn=True)  # [B,Lq,Lk] (已平均 heads)
        cls2all = attn_map[0, 0] if ADD_CLS_TOKEN else attn_map[0].mean(0)     # [3N]
        N = SPATIAL_SIZE * SPATIAL_SIZE
        a_img   = norm_np(cls2all[:N].reshape(SPATIAL_SIZE, SPATIAL_SIZE).cpu().numpy())
        a_recon = norm_np(cls2all[N:2*N].reshape(SPATIAL_SIZE, SPATIAL_SIZE).cpu().numpy())
        a_noise = norm_np(cls2all[2*N:3*N].reshape(SPATIAL_SIZE, SPATIAL_SIZE).cpu().numpy())

        vis_cma_img   = overlay_heatmap(img_np, a_img,   "CMA Image")
        vis_cma_recon = overlay_heatmap(img_np, a_recon, "CMA Recon")
        vis_cma_noise = overlay_heatmap(img_np, a_noise, "CMA Noise")

    # 4) Grad-CAM：对“最终 logit（融合后）”回溯到 feat_gn 的第1次输出（img 分支）
    acts.clear(); grads.clear(); call_counter["feat_gn"] = 0
    difc.zero_grad(set_to_none=True)

    # 构造最小轨迹（T=1），与 DIFCNet 前向接口一致
    with torch.enable_grad():
        img4   = to4(img_t).detach().requires_grad_(True)
        recon4 = to4(recon_t).detach().requires_grad_(True)
        noise4 = to4(noise_t).detach().requires_grad_(True)

        img_traj   = img4.unsqueeze(1)     # [B,1,4,H,W]
        recon_traj = recon4.unsqueeze(1)
        noise_traj = noise4.unsqueeze(1)

        logits = difc(img_traj, recon_traj, noise_traj)     # [B,1]
        score  = logits[:, 0].mean()
        score.backward(retain_graph=False)

    # 计算 Grad-CAM
    A = acts[-1].detach()    # [B,C,H,W]
    G = grads[-1].detach()   # [B,C,H,W]
    w = G.mean(dim=(2,3), keepdim=True)
    cam = F.relu((w * A).sum(dim=1, keepdim=True))          # [B,1,H,W]
    cam = cam / (cam.max() + 1e-8)
    heat = cam[0,0].cpu().numpy()
    vis_gradcam = overlay_heatmap(img_np, heat, "Grad-CAM")

    # 5) 保存四联图
    save_path = os.path.join(SAVE_DIR, f"{name}_quad.png")
    save_four_panel(
        img_np, vis_cma_img, vis_cma_recon, vis_cma_noise, vis_gradcam,
        save_path, suptitle=name
    )

# 6) 清理 hook
handle_f.remove(); handle_b.remove()
print(f"✅ Saved to: {SAVE_DIR}")









# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from cross_modal_attention import CrossModalAttention
# from difc_net import DIFCNet

# # ======================================
# # 配置区
# # ======================================
# device = "cuda" if torch.cuda.is_available() else "cpu"

# FEAT_DIM = 256        # 与 DIFC-Net 的 proj_dim 一致
# SPATIAL_SIZE = 16
# INPUT_SIZE = 256

# img_dir    = "cma_sample/inputs/img"
# recon_dir  = "cma_sample/inputs/recon"
# noise_dir  = "cma_sample/inputs/noise"
# save_dir   = "cma_sample/outputs_ours/heatmaps"
# os.makedirs(save_dir, exist_ok=True)

# # ======================================
# # 工具函数
# # ======================================
# class ToFourChannels(nn.Module):
#     def __init__(self, mode="gray"):
#         super().__init__()
#         self.mode = mode
#     def forward(self, x):  # x: [B,3,H,W]
#         if x.size(1) == 4:
#             return x
#         r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
#         if self.mode == "gray":
#             y = 0.2989*r + 0.5870*g + 0.1140*b
#         elif self.mode == "zeros":
#             y = torch.zeros_like(r)
#         else:
#             y = torch.ones_like(r)
#         return torch.cat([x, y], dim=1)

# def load_image(path, size=INPUT_SIZE):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (size, size))
#     ten = torch.from_numpy(img).permute(2,0,1).float()/255.0
#     return ten.unsqueeze(0).to(device), img

# def normalize_np(x):
#     x = x - x.min()
#     return x / (x.max() + 1e-8)

# def overlay_heatmap(orig, attn, title, save_path):
#     h, w = orig.shape[:2]
#     attn = cv2.resize(attn, (w, h))
#     heat = cv2.applyColorMap((attn*255).astype(np.uint8), cv2.COLORMAP_JET)
#     overlay = (0.5*orig + 0.5*heat).astype(np.uint8)
#     plt.figure(figsize=(4,4))
#     plt.imshow(overlay)
#     plt.axis("off")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=400, bbox_inches="tight")
#     plt.close()

# # ======================================
# # 模型初始化
# # ======================================
# print("🧠 Using DIFC-Net visual backbone...")
# difc = DIFCNet().to(device).eval()
# to4 = ToFourChannels().to(device).eval()

# cma = CrossModalAttention(
#     feat_dim=FEAT_DIM, num_heads=4, spatial_size=SPATIAL_SIZE, add_class_token=True
# ).to(device)
# cma.eval()

# # ======================================
# # 核心特征提取函数
# # ======================================
# def extract_features(img_t, recon_t, noise_t):
#     # 1. 通道扩展
#     img4   = to4(img_t)
#     recon4 = to4(recon_t)
#     noise4 = to4(noise_t)
#     # 2. 投影 + 归一化
#     img_proj   = difc.feat_gn(difc.feat_proj(img4))
#     recon_proj = difc.feat_gn(difc.feat_proj(recon4))
#     noise_proj = difc.feat_gn(difc.feat_proj(noise4))
#     # 3. RAC 特征增强
#     _ = difc.rac(img_proj, recon_proj, noise_proj)
#     # 返回增强后的特征
#     return img_proj, recon_proj, noise_proj


# # ======================================
# # 主循环
# # ======================================
# img_list = sorted(os.listdir(img_dir))
# print(f"🔍 Found {len(img_list)} samples")

# for fname in tqdm(img_list, desc="Generating CMA heatmaps"):
#     name = os.path.splitext(fname)[0]
#     img_path   = os.path.join(img_dir, fname)
#     recon_path = os.path.join(recon_dir, fname)
#     noise_path = os.path.join(noise_dir, fname)
#     if not (os.path.exists(recon_path) and os.path.exists(noise_path)):
#         print(f"⚠️ Skipped {fname}: missing recon/noise.")
#         continue

#     img_t, img_np     = load_image(img_path)
#     recon_t, recon_np = load_image(recon_path)
#     noise_t, noise_np = load_image(noise_path)

#     with torch.no_grad():
#         img_feat, recon_feat, noise_feat = extract_features(img_t, recon_t, noise_t)
#         _, attn_map = cma(img_feat, recon_feat, noise_feat, return_attn=True)
#         cls_to_all = attn_map[0, 0]

#     N = SPATIAL_SIZE * SPATIAL_SIZE
#     img_attn   = normalize_np(cls_to_all[:N].reshape(SPATIAL_SIZE, SPATIAL_SIZE).cpu().numpy())
#     recon_attn = normalize_np(cls_to_all[N:2*N].reshape(SPATIAL_SIZE, SPATIAL_SIZE).cpu().numpy())
#     noise_attn = normalize_np(cls_to_all[2*N:3*N].reshape(SPATIAL_SIZE, SPATIAL_SIZE).cpu().numpy())
#     fused_attn = normalize_np((img_attn + recon_attn + noise_attn) / 3.0)

#     overlay_heatmap(img_np, img_attn,   "CMA Image Attention",   os.path.join(save_dir, f"{name}_img_attn.png"))
#     overlay_heatmap(img_np, recon_attn, "CMA Recon Attention",   os.path.join(save_dir, f"{name}_recon_attn.png"))
#     overlay_heatmap(img_np, noise_attn, "CMA Noise Attention",   os.path.join(save_dir, f"{name}_noise_attn.png"))
#     overlay_heatmap(img_np, fused_attn, "CMA Fused Attention",   os.path.join(save_dir, f"{name}_fused_attn.png"))

# print(f"✅ All CMA attention heatmaps saved in: {save_dir}")
