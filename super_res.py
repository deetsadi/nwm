import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import os
# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

folder = "/home/aditya/nwm/logs/nwm_cdit_l"
gt_folder = f"{folder}/gt"
pred_folder = f"{folder}/pred"
# # let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))

prompt = "a robot doing a task"

for i in range(len(os.listdir(gt_folder))):
    img = Image.open(os.path.join(gt_folder, f"gt_{i}.png"))
    img = img.convert('RGB')

    upscaled_image = pipeline(prompt=prompt, image=img).images[0]
    upscaled_image.save(f"{gt_folder}/gt_{i}_upscaled.png")

for i in range(len(os.listdir(pred_folder))):
    img = Image.open(os.path.join(pred_folder, f"pred_{i}.png"))
    img = img.convert('RGB')

    upscaled_image = pipeline(prompt=prompt, image=img).images[0]
    upscaled_image.save(f"{pred_folder}/pred_{i}_upscaled.png")