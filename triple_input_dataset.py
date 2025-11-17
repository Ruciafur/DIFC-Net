import os
import torch
from torch.utils.data import Dataset

class TrajectoryCacheDataset(Dataset):
    def __init__(self, cache_root):
        self.samples = []
        for c in ['fake', 'real']:
            cache_dir = os.path.join(cache_root, c)
            for f in sorted(os.listdir(cache_dir)):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(cache_dir, f), 0 if c == 'fake' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label = self.samples[idx]
        # 1. 加载 pt 文件
        data = torch.load(pt_path)
        # 支持不同保存格式，推荐统一保存为 {"traj":..., "label":...}
        # 兼容直接保存的tensor情况
        if isinstance(data, dict):
            traj = data["traj"]    # [3, T, C, H, W]
        else:
            traj = data            # [3, T, C, H, W]
        # 2. 拆分为三路（img, recon, noise），并确保float32
        img_traj   = traj[0].float()    # [T, C, H, W]
        recon_traj = traj[1].float()
        noise_traj = traj[2].float()
        # 3. 返回四元组
        return img_traj, recon_traj, noise_traj, torch.tensor(label, dtype=torch.float32)





# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as T

# class TripleInputDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform or T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor()
#         ])
#         self.samples = []
#         for label, category in enumerate(["real", "fake"]):
#             img_dir = os.path.join(root_dir, category, "img")
#             recon_dir = os.path.join(root_dir, category, "recon")
#             noise_dir = os.path.join(root_dir, category, "noise")

#             img_files = set(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
#             recon_files = set(f for f in os.listdir(recon_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
#             noise_files = set(f for f in os.listdir(noise_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
#             common_files = img_files & recon_files & noise_files
#             if len(common_files) == 0:
#                 print(f"警告: {category} 类别下未找到三路匹配的图片文件。")
#             for fname in sorted(common_files):
#                 sample = {
#                     "img": os.path.join(img_dir, fname),
#                     "recon": os.path.join(recon_dir, fname),
#                     "noise": os.path.join(noise_dir, fname),
#                     "label": label
#                 }
#                 self.samples.append(sample)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         img = Image.open(sample["img"]).convert("RGB")
#         recon = Image.open(sample["recon"]).convert("RGB")
#         noise = Image.open(sample["noise"]).convert("RGB")
#         img = self.transform(img)
#         recon = self.transform(recon)
#         noise = self.transform(noise)
#         label = sample["label"]
#         return img, recon, noise, label