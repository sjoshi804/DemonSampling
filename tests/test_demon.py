import torch
import os
import numpy as np
from transformers import AutoModel, AutoProcessor
from datetime import datetime

# Local application/library specific imports
from api import demon_sampling, get_init_latent
from utils import from_latent_to_pil


pickscore_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pickscore_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to('cuda')

prompt = "An astronaut riding a horse on Mars."
condition = {
    "prompts": [prompt],
    "cfgs": [2]
}

if not os.path.exists('./tmp/images'):
    os.makedirs('./tmp/images')


@torch.inference_mode()
def pickscore_reward(latents):
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    pils = from_latent_to_pil(latents)
    for i, pil in enumerate(pils):
        pil.save(f'./tmp/images/{datetime_str}_{i}.png')

    inputs = pickscore_processor(images=pils, text=prompt, return_tensors="pt", padding=True).to('cuda')
    return pickscore_model(**inputs).logits_per_image.squeeze().cpu().numpy().tolist()

def test_demon():
    x = get_init_latent()
    x = demon_sampling(
        x, 
        condition,
        pickscore_reward, 
        beta=0.125, 
        tau="adaptive", 
        K=16,
        T=64, 
        demon_type="tanh",
        r_of_c="baseline",
        log_dir='./tmp')
    pil = from_latent_to_pil(x)
    pil.save('./tmp/demon_sampling.png')
    assert os.path.exists('./tmp/demon_sampling.png')

# Run the following comman to test the function:
# time pytest tests/test_demon.py