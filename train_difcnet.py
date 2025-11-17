#v3
# train_difcnet.py
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from triple_input_dataset import TrajectoryCacheDataset
from difc_net import DIFCNet
from losses import LabelSmoothingBCELoss

# ========= 画图（无显示环境） =========
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------ 可配置超参 ------------------------
BATCH_SIZE   = 8
NUM_EPOCHS   = 30
BASE_LR      = 1e-4
ACCUM_STEPS  = 4            # 梯度累积步数（8*4=32 的等效批大小）
MAX_NORM     = 1.0          # 梯度裁剪阈值，None/0 关闭
USE_AMP      = True         # 混合精度开关
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR    = "cache/dalle3/train"    # 轨迹缓存root
SAVE_DIR     = "runs_dalle3"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED         = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

# ------------------------ 数据集加载 ------------------------
dataset = TrajectoryCacheDataset(CACHE_DIR)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    num_workers=2, pin_memory=True, persistent_workers=True
)

# ------------------------ 模型与优化器 ------------------------
model = DIFCNet(proj_dim=256).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
criterion = LabelSmoothingBCELoss(eps=0.1)  # 你当前使用的损失

# 学习率调度器（基于 Val AUC）
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

# AMP 工具
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ------------------------ 训练/验证 函数 ------------------------
def current_lr(optim):
    return float(optim.param_groups[0]["lr"])

def train_one_epoch(loader, epoch):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Training {epoch}")
    for step, (img_traj, recon_traj, noise_traj, label) in pbar:
        img_traj   = img_traj.to(DEVICE, non_blocking=True)  # [B, T, 4, 64, 64]
        recon_traj = recon_traj.to(DEVICE, non_blocking=True)
        noise_traj = noise_traj.to(DEVICE, non_blocking=True)
        label      = label.float().to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(img_traj, recon_traj, noise_traj).view(-1)
            loss = criterion(logits, label)
            # 累积
            loss_scaled = loss / ACCUM_STEPS

        # backward (AMP)
        scaler.scale(loss_scaled).backward()

        # 每 ACCUM_STEPS 更新一次
        if (step + 1) % ACCUM_STEPS == 0:
            # 反缩放 + 可选梯度裁剪
            if MAX_NORM and MAX_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())

        with torch.no_grad():
            all_labels.extend(label.detach().cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

        if step % 20 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr(optimizer):.2e}"
            })

    # 处理最后不足 ACCUM_STEPS 的残留 batch
    # （如果循环里最后一次没有 step()，通常在下一 epoch 才会再 step。一般可以忽略）
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return total_loss / len(loader), auc

@torch.no_grad()
def validate(loader, epoch):
    model.eval()
    all_labels, all_preds = [], []
    pbar = tqdm(loader, desc=f"Validation {epoch}")
    for img_traj, recon_traj, noise_traj, label in pbar:
        img_traj   = img_traj.to(DEVICE, non_blocking=True)
        recon_traj = recon_traj.to(DEVICE, non_blocking=True)
        noise_traj = noise_traj.to(DEVICE, non_blocking=True)
        label      = label.float().to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(img_traj, recon_traj, noise_traj).view(-1)

        all_labels.extend(label.detach().cpu().numpy())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    return auc

# ------------------------ 训练主循环 & 记录 ------------------------
epoch_list, train_auc_hist, val_auc_hist, train_loss_hist, lr_hist = [], [], [], [], []
best_auc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_auc = train_one_epoch(train_loader, epoch)
    val_auc = validate(val_loader, epoch)

    # 调度器更新（监控 Val AUC）
    scheduler.step(val_auc)
    cur_lr = current_lr(optimizer)

    # 记录历史
    epoch_list.append(epoch)
    train_auc_hist.append(train_auc)
    val_auc_hist.append(val_auc)
    train_loss_hist.append(train_loss)
    lr_hist.append(cur_lr)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
          f"Val AUC: {val_auc:.4f} | LR: {cur_lr:.8f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "difcnet_best.pth"))
        print(f"✅ Best model saved with Val AUC: {best_auc:.4f}")

# ------------------------ 保存 CSV 与曲线 ------------------------
csv_path = os.path.join(SAVE_DIR, "metrics_history.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_auc", "val_auc", "lr"])
    for e, l, ta, va, lr in zip(epoch_list, train_loss_hist, train_auc_hist, val_auc_hist, lr_hist):
        writer.writerow([e, l, ta, va, lr])
print(f"📄 Metrics saved to {csv_path}")

# AUC 曲线
plt.figure(figsize=(7.5, 5.5))
plt.plot(epoch_list, train_auc_hist, label="Train AUC", color="#1f77b4", linewidth=2)
plt.plot(epoch_list, val_auc_hist, label="Val AUC", color="#ff7f0e", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("AUC Curve")
plt.grid(True, alpha=0.3)
plt.legend()
auc_png = os.path.join(SAVE_DIR, "auc_curve.png")
plt.savefig(auc_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"📈 AUC curve saved to {auc_png}")

# 学习率曲线（对数尺度）
plt.figure(figsize=(7.5, 5.5))
plt.plot(epoch_list, lr_hist, label="Learning Rate", color="#2ca02c", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("LR (log scale)")
plt.title("Learning Rate Curve")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend()
lr_png = os.path.join(SAVE_DIR, "lr_curve.png")
plt.savefig(lr_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"📉 LR curve saved to {lr_png}")



#v2
# import os
# import csv
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from triple_input_dataset import TrajectoryCacheDataset
# from difc_net import DIFCNet
# from sklearn.metrics import roc_auc_score
# from tqdm import tqdm
# from losses import LabelSmoothingBCELoss

# # ========= 画图（无显示环境） =========
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# # 配置参数
# BATCH_SIZE = 8
# NUM_EPOCHS = 30
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CACHE_DIR = "cache/train"  # 指向.pt缓存目录
# SAVE_DIR = "runs_v3"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # 1. 数据集加载
# dataset = TrajectoryCacheDataset(CACHE_DIR)
# val_size = int(0.1 * len(dataset))
# train_size = len(dataset) - val_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

# # 2. 模型实例化
# model = DIFCNet(proj_dim=256).to(DEVICE)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# # criterion = nn.BCEWithLogitsLoss()
# criterion = LabelSmoothingBCELoss(eps=0.1)
# # 3. 学习率调度器（基于 Val AUC）
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

# # 4. 历史记录
# epoch_list = []
# train_auc_hist, val_auc_hist, train_loss_hist, lr_hist = [], [], [], []

# def current_lr(optim):
#     return optim.param_groups[0]["lr"]

# def train_one_epoch(loader):
#     model.train()
#     total_loss = 0.0
#     all_labels, all_preds = [], []

#     for img_traj, recon_traj, noise_traj, label in tqdm(loader, desc="Training"):
#         # img_traj, recon_traj, noise_traj: [B, T, C, H, W]
#         img_traj   = img_traj.to(DEVICE)
#         recon_traj = recon_traj.to(DEVICE)
#         noise_traj = noise_traj.to(DEVICE)
#         label      = label.float().to(DEVICE)

#         logits = model(img_traj, recon_traj, noise_traj).view(-1)
#         loss = criterion(logits, label)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         all_labels.extend(label.detach().cpu().numpy())
#         all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

#     auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
#     return total_loss / len(loader), auc

# @torch.no_grad()
# def validate(loader):
#     model.eval()
#     all_labels, all_preds = [], []
#     for img_traj, recon_traj, noise_traj, label in tqdm(loader, desc="Validation"):
#         img_traj   = img_traj.to(DEVICE)
#         recon_traj = recon_traj.to(DEVICE)
#         noise_traj = noise_traj.to(DEVICE)
#         label      = label.float().to(DEVICE)

#         logits = model(img_traj, recon_traj, noise_traj).view(-1)
#         all_labels.extend(label.detach().cpu().numpy())
#         all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())

#     auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
#     return auc

# best_auc = 0.0
# for epoch in range(1, NUM_EPOCHS + 1):
#     train_loss, train_auc = train_one_epoch(train_loader)
#     val_auc = validate(val_loader)

#     # 调度器更新
#     scheduler.step(val_auc)
#     cur_lr = current_lr(optimizer)

#     # 记录历史
#     epoch_list.append(epoch)
#     train_auc_hist.append(train_auc)
#     val_auc_hist.append(val_auc)
#     train_loss_hist.append(train_loss)
#     lr_hist.append(cur_lr)

#     print(f"Epoch {epoch}/{NUM_EPOCHS} | "
#           f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | "
#           f"Val AUC: {val_auc:.4f} | LR: {cur_lr:.8f}")

#     if val_auc > best_auc:
#         best_auc = val_auc
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, "difcnet_best.pth"))
#         print(f"✅ Best model saved with Val AUC: {best_auc:.4f}")

# # 5. 保存历史 CSV
# csv_path = os.path.join(SAVE_DIR, "metrics_history.csv")
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["epoch", "train_loss", "train_auc", "val_auc", "lr"])
#     for e, l, ta, va, lr in zip(epoch_list, train_loss_hist, train_auc_hist, val_auc_hist, lr_hist):
#         writer.writerow([e, l, ta, va, lr])
# print(f"📄 Metrics saved to {csv_path}")

# # 6. 绘制 AUC 曲线
# plt.figure(figsize=(7, 5))
# plt.plot(epoch_list, train_auc_hist, label="Train AUC", color="#1f77b4", linewidth=2)
# plt.plot(epoch_list, val_auc_hist, label="Val AUC", color="#ff7f0e", linewidth=2)
# plt.xlabel("Epoch")
# plt.ylabel("AUC")
# plt.title("AUC Curve")
# plt.grid(True, alpha=0.3)
# plt.legend()
# auc_png = os.path.join(SAVE_DIR, "auc_curve.png")
# plt.savefig(auc_png, dpi=150, bbox_inches="tight")
# plt.close()
# print(f"📈 AUC curve saved to {auc_png}")

# # 7. 绘制学习率曲线（对数尺度更直观）
# plt.figure(figsize=(7, 5))
# plt.plot(epoch_list, lr_hist, label="Learning Rate", color="#2ca02c", linewidth=2)
# plt.yscale("log")
# plt.xlabel("Epoch")
# plt.ylabel("LR (log scale)")
# plt.title("Learning Rate Curve")
# plt.grid(True, which="both", linestyle="--", alpha=0.3)
# plt.legend()
# lr_png = os.path.join(SAVE_DIR, "lr_curve.png")
# plt.savefig(lr_png, dpi=150, bbox_inches="tight")
# plt.close()
# print(f"📉 LR curve saved to {lr_png}")