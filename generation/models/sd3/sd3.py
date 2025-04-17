import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm
from models.prompt_loader import get_txt_files
import sys
import os


def sd3_t2i(prompts, image_names):

    dir_path = './results'
    os.makedirs(dir_path, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, device_map = 'balanced')
    # pipe = pipe.to("cuda")

    for p_idx, prompt in tqdm(enumerate(prompts), total = len(prompts)):

        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=100,
            guidance_scale=15.0,
            height = 1024,
            width = 1024,
        ).images[0]

        image.save(f"{dir_path}/{image_names[p_idx]}.png")
        with open(f"{dir_path}/{image_names[p_idx]}.txt", 'w') as f:
            f.write(prompt)
            f.write('\n')
    
if __name__ == "__main__":

    dir_path = sys.argv[1]


    prompts, names = get_txt_files(f'../{dir_path}')

    sd3_t2i(prompts, names)
