import torch
from diffusers import FluxPipeline
from models.prompt_loader import get_txt_files
from accelerate import infer_auto_device_map
from tqdm import tqdm
import sys
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"

def flux_t2i(prompts, image_names):

    model_path = "/home/yongqian/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="balanced")

    dir_path = './results'
    os.makedirs(dir_path, exist_ok=True)

    for p_idx, prompt in tqdm(enumerate(prompts), total = len(prompts)):

        # Run inference
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=100,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        image.save(f"{dir_path}/{image_names[p_idx]}.png")
        with open(f"{dir_path}/{image_names[p_idx]}.txt", 'w') as f:
            f.write(prompt)
            f.write('\n')

if __name__ == "__main__":
    dir_path = sys.argv[1]

    prompts, names = get_txt_files(f'../{dir_path}')

    flux_t2i(prompts, names)
