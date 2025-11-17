# import os
# import torch
# from PIL import Image
# from tqdm import tqdm
# from torchvision import transforms
# from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
# from transformers import BlipProcessor, BlipForConditionalGeneration
 
# # ======= 环境准备 =======
# device = "cuda" if torch.cuda.is_available() else "cpu"
 
# # ======= 模型加载 =======
# pipe = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     torch_dtype=torch.float16
# ).to(device)
 
# # 设置 DDIM 反向调度器
# pipe.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
#     "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
# )
# pipe.inverse_scheduler.set_timesteps(50)
 
# vae = pipe.vae
# unet = pipe.unet
# tokenizer = pipe.tokenizer
# text_encoder = pipe.text_encoder
 
# # Caption 模型
# blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
 
# # 图像预处理
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])
 
# @torch.no_grad()
# def caption_image(image):
#     inputs = blip_processor(images=image, return_tensors="pt").to(device)
#     output = blip.generate(**inputs)
#     return blip_processor.decode(output[0], skip_special_tokens=True)
 
# @torch.no_grad()
# def save_image(tensor, path):
#     img = (tensor.clamp(-1, 1) + 1) / 2
#     arr = (img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#     Image.fromarray(arr).save(path)
 
# @torch.no_grad()
# def invert_and_reconstruct(input_dir, noise_dir, recon_dir):
#     os.makedirs(noise_dir, exist_ok=True)
#     os.makedirs(recon_dir, exist_ok=True)
 
#     for fname in tqdm(os.listdir(input_dir)):
#         if not fname.lower().endswith((".jpg", ".png")):
#             continue
 
#         image_path = os.path.join(input_dir, fname)
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = transform(image).unsqueeze(0).to(device).half()
 
#         # 自动生成 prompt
#         caption = caption_image(image)
#         prompt_ids = tokenizer(caption, return_tensors="pt").input_ids.to(device)
#         prompt_emb = text_encoder(prompt_ids)[0].half()
 
#         # 编码图像 latent z_0
#         z0_dist = vae.encode(image_tensor)
#         z0 = z0_dist.latent_dist.sample() * 0.18215  # scale factor for SD 1.4
 
#         # ======== 使用 DDIMInverseScheduler 获取 z_T ========
#         timesteps = pipe.inverse_scheduler.timesteps
#         zt = z0.clone()
#         for t in reversed(timesteps):
#             # DDIM 反向过程
#             noise_pred = unet(zt, t, encoder_hidden_states=prompt_emb).sample
#             zt = pipe.inverse_scheduler.step(
#                 model_output=noise_pred,
#                 timestep=t,
#                 sample=zt
#             ).prev_sample
 
#         # 解码 z_T 得到 noise-like 图像
#         x_noise = vae.decode(zt / 0.18215).sample
#         save_image(x_noise, os.path.join(noise_dir, fname))
 
#         # 使用 forward scheduler 还原图像 latent
#         pipe.scheduler.set_timesteps(50)
#         z_recon = zt.clone()
#         for t in pipe.scheduler.timesteps:
#             noise_pred = unet(z_recon, t, encoder_hidden_states=prompt_emb).sample
#             z_recon = pipe.scheduler.step(
#                 model_output=noise_pred,
#                 timestep=t,
#                 sample=z_recon
#             ).prev_sample
 
#         x_recon = vae.decode(z_recon / 0.18215).sample
#         save_image(x_recon, os.path.join(recon_dir, fname))
 
# # ======= 执行流程 =======
# if __name__ == "__main__":
#     invert_and_reconstruct(
#         input_dir="dataset/test/sd15/img",       # 原始图像路径
#         noise_dir="dataset/test/sd15/noise",     # z_T 的解码图像
#         recon_dir="dataset/test/sd15/recon"      # 重建图像
#     )
    


    
    
import os
import sys
import traceback
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration

# ======= 环境准备 =======
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
if device == "cuda":
    torch.backends.cudnn.benchmark = True

# ======= 工具函数 =======
def to_pil_rgb(img):
    """
    把各种可能的输入（PIL / numpy / torch.Tensor）统一为 PIL RGB。
    并在必要时把单通道扩成三通道、把 (C,H,W) 转为 (H,W,C)、把浮点[0,1]转为uint8。
    """
    from PIL import Image
    import numpy as np

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    # 尽量处理 numpy / torch 情况
    try:
        import numpy as np
        if torch.is_tensor(img):
            x = img.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3, 4):  # (C,H,W) -> (H,W,C)
                if x.shape[0] == 1:
                    x = x.expand(3, -1, -1)
                x = x.permute(1, 2, 0).contiguous().numpy()
            else:
                x = x.numpy()
            img = x  # 继续当作 numpy 处理

        if isinstance(img, np.ndarray):
            x = img
            if x.ndim == 2:  # (H,W) -> (H,W,1) -> (H,W,3)
                x = np.stack([x] * 3, axis=-1)
            if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
                x = np.transpose(x, (1, 2, 0))
            if x.ndim == 3 and x.shape[-1] == 1:
                x = np.repeat(x, 3, axis=-1)
            # 类型/范围修正
            if x.dtype != np.uint8:
                if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
                    x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    x = np.clip(x, 0, 255).astype(np.uint8)
            return Image.fromarray(x, mode="RGB")
    except Exception:
        pass

    raise TypeError(f"Unsupported image type for to_pil_rgb(): {type(img)}")


def is_too_small(pil_img: Image.Image, min_edge: int = 4) -> bool:
    w, h = pil_img.size
    return (w < min_edge) or (h < min_edge)


def safe_open_image(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()  # 强制读入，尽早发现损坏
        return img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        raise


@torch.no_grad()
def save_image(tensor: torch.Tensor, path: str):
    """
    将范围[-1,1]或[0,1]的张量保存为PNG/JPG。自动处理 (B,C,H,W)/(C,H,W)/(H,W,C)。
    """
    import numpy as np

    x = tensor
    if torch.is_tensor(x):
        x = x.detach().cpu()

    if x.ndim == 4:      # (B,C,H,W) 取第一张
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):  # (C,H,W) -> (H,W,C)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        x = x.permute(1, 2, 0)
    x = x.contiguous().numpy()

    # 归一化到[0,1]
    x_min, x_max = x.min(), x.max()
    if x_min < -1.01 or x_max > 1.01:
        # 如果范围不在[-1,1]附近，尝试按[0,1]裁剪
        x = np.clip(x, 0.0, 1.0)
    else:
        x = (x.clip(-1, 1) + 1) / 2.0

    x = (x * 255.0).round().clip(0, 255).astype("uint8")
    Image.fromarray(x).save(path)


# ======= 模型加载 =======
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 设置 DDIM 反向调度器
pipe.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)
pipe.inverse_scheduler.set_timesteps(50)

vae = pipe.vae
unet = pipe.unet
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

# Caption 模型（BLIP）
blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# 图像预处理（3通道）
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


@torch.no_grad()
def caption_image(image):
    """
    更健壮的 caption：先统一成 PIL RGB，并过滤过小图像。
    出错时退回一个保底 prompt，避免打断整批处理。
    """
    try:
        pil_img = to_pil_rgb(image)
        if is_too_small(pil_img, min_edge=4):
            # 对于过小图片，放大到最小合格尺寸再送处理器（或直接跳过）
            pil_img = pil_img.resize((4, 4), Image.BICUBIC)

        inputs = blip_processor(images=pil_img, return_tensors="pt").to(device)
        output = blip.generate(**inputs, max_new_tokens=64)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
        caption = caption.strip()
        if not caption:
            caption = "a photo"
        return caption
    except Exception:
        # 打印错误但不中断
        print("[warn] caption_image failed, fallback to default caption.", file=sys.stderr)
        traceback.print_exc()
        return "a photo"


@torch.no_grad()
def invert_and_reconstruct(input_dir, noise_dir, recon_dir):
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    files.sort()

    pbar = tqdm(files, desc="Processing")
    for fname in pbar:
        image_path = os.path.join(input_dir, fname)

        # 1) 读取原图（健壮）
        try:
            image = safe_open_image(image_path)  # PIL RGB
        except Exception:
            print(f"[error] cannot open image: {image_path}", file=sys.stderr)
            continue

        # 2) 图像 -> tensor（用于VAE编码）
        image_tensor = transform(image).unsqueeze(0).to(device)
        if device == "cuda":
            image_tensor = image_tensor.half()

        # 3) 自动生成 prompt（健壮）
        caption = caption_image(image)

        # 4) 文本嵌入
        prompt_ids = tokenizer(caption, return_tensors="pt").input_ids.to(device)
        prompt_emb = text_encoder(prompt_ids)[0]
        if device == "cuda":
            prompt_emb = prompt_emb.half()

        # 5) 编码图像 latent z0
        z0_dist = vae.encode(image_tensor)
        # diffusers 的 VAE.encode 结果兼容 .latent_dist.sample() / .sample
        if hasattr(z0_dist, "latent_dist"):
            z0 = z0_dist.latent_dist.sample()
        else:
            z0 = z0_dist.sample
        z0 = z0 * 0.18215  # scale for SD 1.4

        # 6) 使用 DDIMInverseScheduler 反向获得 zT
        timesteps = pipe.inverse_scheduler.timesteps
        zt = z0.clone()

        # 用 autocast 提升半精度运行稳定性
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else torch.cpu.amp.autocast(enabled=False)
        with autocast_ctx:
            for t in reversed(timesteps):
                noise_pred = unet(zt, t, encoder_hidden_states=prompt_emb).sample
                zt = pipe.inverse_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=zt
                ).prev_sample

        # 7) 解码 zT（noise-like 图）
        x_noise = vae.decode(zt / 0.18215).sample
        save_image(x_noise, os.path.join(noise_dir, fname))

        # 8) 用 forward scheduler 还原图像 latent
        pipe.scheduler.set_timesteps(50)
        z_recon = zt.clone()
        with autocast_ctx:
            for t in pipe.scheduler.timesteps:
                noise_pred = unet(z_recon, t, encoder_hidden_states=prompt_emb).sample
                z_recon = pipe.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=z_recon
                ).prev_sample

        # 9) 解码重建图
        x_recon = vae.decode(z_recon / 0.18215).sample
        save_image(x_recon, os.path.join(recon_dir, fname))

        pbar.set_postfix_str(f"caption='{caption[:40]}'")

# ======= 执行流程 =======
if __name__ == "__main__":
    invert_and_reconstruct(
        input_dir="dataset/train/real/img",       # 原始图像路径
        noise_dir="dataset/train/real/noise",     # z_T 的解码图像
        recon_dir="dataset/train/real/recon"      # 重建图像
    )
