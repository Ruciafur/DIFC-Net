# evaluate_difcnet.py
import os, json
import torch
import numpy as np
from torch.utils.data import DataLoader
from triple_input_dataset import TrajectoryCacheDataset
from difc_net import DIFCNet
from tqdm import tqdm
from metrics_utils import evaluate_binary, plot_roc_pr_det, reliability_diagram

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_CACHE = "cache/test/sdxl"
SAVE_DIR   = "eval_report/sdxl"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1) 数据集
test_dataset = TrajectoryCacheDataset(TEST_CACHE)
test_loader  = DataLoader(test_dataset, batch_size=8, num_workers=2)

# 2) 模型
model = DIFCNet(proj_dim=256).to(DEVICE)
model.load_state_dict(torch.load("runs_v3/difcnet_best.pth", map_location=DEVICE))
model.eval()

# 3) 推理收集
y_true_list, y_score_list = [], []
with torch.no_grad():
    for img_traj, recon_traj, noise_traj, label in tqdm(test_loader, desc="Testing"):
        img_traj   = img_traj.to(DEVICE)   # [B, T, 4, 64, 64]
        recon_traj = recon_traj.to(DEVICE)
        noise_traj = noise_traj.to(DEVICE)
        logits = model(img_traj, recon_traj, noise_traj).view(-1)  # [B]
        y_true_list.append(label.numpy())
        y_score_list.append(logits.detach().cpu().numpy())

y_true  = np.concatenate(y_true_list, axis=0).astype(np.int32)  # 0/1
y_score = np.concatenate(y_score_list, axis=0)                  # logits

# 4) 评估与出图
metrics = evaluate_binary(y_true, y_score, thr=None)  # 会自动用Youden阈值
with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("== Metrics ==")
print(json.dumps(metrics, indent=2))

plot_roc_pr_det(y_true, y_score, SAVE_DIR)
reliability_diagram(y_true, y_score, os.path.join(SAVE_DIR, "reliability.png"))

print(f"All figures & metrics saved to: {SAVE_DIR}")
