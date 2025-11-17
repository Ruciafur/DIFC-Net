import os
import torch
from tqdm import tqdm
from ltt import LatentTrajectoryTracker  # 你的LatentTrajectoryTracker类

device = "cuda" if torch.cuda.is_available() else "cpu"
ltt = LatentTrajectoryTracker(device=device)

def process_folder(split, category, img_dir, recon_dir, noise_dir, out_dir, prompt="A photo", overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    fnames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    for fname in tqdm(fnames, desc=f"{split}/{category}"):
        img_path = os.path.join(img_dir, fname)
        recon_path = os.path.join(recon_dir, fname)
        noise_path = os.path.join(noise_dir, fname)
        cache_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".pt")
        if not overwrite and os.path.exists(cache_path):
            continue
        from PIL import Image
        from torchvision import transforms
        pil2tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        img = pil2tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        recon = pil2tensor(Image.open(recon_path).convert("RGB")).unsqueeze(0).to(device)
        noise = pil2tensor(Image.open(noise_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            traj = ltt.track(img, recon, noise, prompt)  # [1, 3, T, C, H, W]
        sample = {
            "traj": traj.squeeze(0).cpu(),          # [3, T, C, H, W]
            "label": 1 if category == "fake" else 0
        }
        torch.save(sample, cache_path)

splits = ["test"]
categories = ["sdxl"]
for split in splits:
    for category in categories:
        base = f"data1/{split}/{category}"
        process_folder(
            split, category,
            img_dir=os.path.join(base, "img"),
            recon_dir=os.path.join(base, "recon"),
            noise_dir=os.path.join(base, "noise"),
            out_dir=f"cache/{split}/{category}",
            prompt="A photo"
        )
        

# splits = ["train", "val", "test"]
# categories = ["real", "fake"]
# for split in splits:
#     for category in categories:
#         base = f"data/{split}/{category}"
#         process_folder(
#             split, category,
#             img_dir=os.path.join(base, "img"),
#             recon_dir=os.path.join(base, "recon"),
#             noise_dir=os.path.join(base, "noise"),
#             out_dir=f"cache/{split}/{category}",
#             prompt="A photo"
#         )
        