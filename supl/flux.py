import torch
from diffusers import FluxPipeline
# from models.prompt_loader import get_txt_files
from accelerate import infer_auto_device_map
from tqdm import tqdm
import sys
import os
import json

def flux_t2i(prompts, image_names, seed):

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    pipe.to('cuda')

    dir_path = f'./results/{seed}'
    os.makedirs(dir_path, exist_ok=True)

    for p_idx, prompt in tqdm(enumerate(prompts), total = len(prompts)):
        try:
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                # guidance_scale=6,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(seed)
            ).images[0]

            # Save the image-prompt pairs
            image.save(f"{dir_path}/{image_names[p_idx]}.png")
            with open(f"{dir_path}/{image_names[p_idx]}.txt", 'w') as f:
                f.write(prompt)
                f.write('\n')
        except Exception as e:
            print(f"Error generating image for prompt {p_idx}: {e}")
            pass

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed = 3407

    file_path = './prompts.json'  # Replace 'data.json' with your actual file name

    # Open and read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize lists to store prompts and names
    prompts = []
    names = []

    # Accessing data
    for idx, item in enumerate(data):
        # Extracting the relevant fields
        wide_topic = item.get("Wide Topic")
        topic = item.get("Topic")
        input_prompt = item.get("inputPrompt")
        output = item.get("output")
        output = output.split('Output')[-1].replace(':','')

        # Constructing the prompt
        prompts.append(output)
        name = topic+ '_'+ str(idx)
        names.append(name.replace('/','_'))
        
    print(f"{len(prompts)} prompts found")
    flux_t2i(prompts, names, int(seed))
