import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline

class LatentTrajectoryTracker:
    def __init__(self, model_id="/openbayes/home/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b", device='cuda'):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

    @torch.no_grad()
    def track(self, img, recon_img, noise_img, prompt, num_steps=50, guidance_scale=7.5):
        batch_size = img.shape[0]
        def process_one(x):
            x = torch.nn.functional.interpolate(
                x, size=(512, 512), mode='bilinear', align_corners=False
            ).to(self.device).half()
            latents = self.vae.encode(x).latent_dist.sample() * 0.18215
            latents = torch.nn.functional.interpolate(latents, size=(64, 64), mode='bilinear', align_corners=False)
            self.inverse_scheduler.set_timesteps(num_steps)
            trajectory = [latents]
            text_inputs = self.tokenizer(
                [prompt] * batch_size, padding="max_length",
                max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            text_embedding = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
            uncond_inputs = self.tokenizer(
                [""] * batch_size, padding="max_length",
                max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embedding = self.text_encoder(uncond_inputs.input_ids.to(self.device))[0]
            text_embedding = torch.cat([uncond_embedding, text_embedding], dim=0)
            cur_latents = latents
            for timestep in self.inverse_scheduler.timesteps:
                latent_model_input = torch.cat([cur_latents] * 2, dim=0)
                noise_pred = self.unet(
                    latent_model_input, timestep,
                    encoder_hidden_states=text_embedding
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                cur_latents = self.inverse_scheduler.step(noise_pred, timestep, cur_latents).prev_sample
                trajectory.append(cur_latents)
            return torch.stack(trajectory, dim=1)   # [B, T+1, C, H, W]
        traj_img   = process_one(img)
        traj_recon = process_one(recon_img)
        traj_noise = process_one(noise_img)
        all_traj = torch.stack([traj_img, traj_recon, traj_noise], dim=1)  # [B, 3, T+1, C, H, W]
        return all_traj.float()  # <--- 保证输出 float32