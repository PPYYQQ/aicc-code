from openai import OpenAI
import os
import requests
import time
import json
from models.prompt_loader import get_txt_files
from tqdm import tqdm

def dalle_t2i(prompt, image_name, api_key):
    api_key = 'YourAPIKey'
    client = OpenAI(api_key=api_key)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    meta_info = response
    image_url = response.data[0].url

    print(f'Image URL: {image_url}')

    # Send HTTP request to download the image
    response = requests.get(image_url)

    # If the request is successful

    dir_path = './results'

    # Check if the directory exists
    if not os.path.exists(dir_path):
        # If it doesn't exist, create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")

    if response.status_code == 200:
        # Save the image locally
        with open(f"{dir_path}/{image_name}.png", "wb") as file:
            file.write(response.content)
        file.close()
        
        # Extract relevant info or convert the response to a string/JSON.
        meta_info_str = json.dumps(meta_info, default=lambda o: o.__dict__, indent=4)
        with open(f"./results/{image_name}.txt", "w") as file:
            file.write(meta_info_str)
        file.close()
        
        print(f"Image successfully downloaded and saved as {image_name}.png")
    else:
        print(f"Download failed, status code: {response.status_code}")

        

if __name__ == '__main__':
    prompt = " A person drinking soda, with each sip causing tiny explosions inside their body, symbolizing gradual self-destruction and health decline"

    prompts, names = get_txt_files('./gen_prompts/cc')

    for idx, prompt in enumerate(tqdm(prompts, total=len(prompts))):
        image_name = names[idx]
        api_key = os.getenv('API_KEY')
        dalle_t2i(prompt, image_name, api_key)

