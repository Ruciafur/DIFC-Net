import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
 
# ====== 参数设置 ======
prompts = [
    "a futuristic city skyline at night",
    "a cute cat wearing a space helmet",
    "an astronaut riding a horse on Mars",
    "a cyberpunk street scene with neon lights",
    "a fantasy landscape with flying islands",
    "a portrait of a robot with human features",
    "a medieval castle in the mountains",
    "a bowl of ramen with floating ingredients",
    "a surreal dreamlike desert with giant mushrooms",
    "a majestic tiger walking in snow",
    "a steampunk airship soaring through the clouds",
    "a glowing jellyfish swimming in a neon-lit ocean",
    "a fantasy forest with glowing trees and mythical creatures",
    "a futuristic race car zooming through a neon city",
    "a dragon flying over an ancient temple",
    "a giant robot standing in a city during sunset",
    "a wizard casting a spell in a dark forest",
    "a futuristic underwater city with glowing coral reefs",
    "a giant robot walking through a post-apocalyptic city",
    "a majestic eagle flying over a mountain range",
    "a robot and human sitting together, watching a sunset",
    "a serene lake with bioluminescent plants and glowing fish",
    "a cyberpunk hacker in a dark room surrounded by holograms",
    "a samurai warrior with a glowing katana in a bamboo forest",
    "a time traveler standing on a futuristic train platform",
    "a vampire in a gothic mansion with candles and cobwebs",
    "a mystical waterfall cascading down into a glowing pool",
    "a surreal landscape with floating clocks and melting mountains",
    "a spaceship exploring an unknown planet with purple skies",
    "a phoenix rising from the ashes in a fiery landscape"
]
 
output_dir = "generated_fake_images"
os.makedirs(output_dir, exist_ok=True)
 
# ====== 加载 Stable Diffusion 模型 ======
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to("cuda")
 
pipe.enable_attention_slicing()
 
# ====== 图像生成 ======
for idx, prompt in enumerate(prompts):
    with torch.autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    save_path = os.path.join(output_dir, f"{idx+1:05}.jpg")
    image.save(save_path)
    print(f"Saved: {save_path} ← {prompt}")