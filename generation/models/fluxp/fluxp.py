import os
import time
from tqdm import tqdm
import requests
import json
import http.client
from models.prompt_loader import get_txt_files
import sys
import os
import requests

bfl_key = 'YourAPIKey'

def fluxp_t2i(prompt):
    request = requests.post(
        'https://api.bfl.ml/v1/flux-pro-1.1',
        headers={
            'accept': 'application/json',
            'x-key': bfl_key,
            'Content-Type': 'application/json',
        },
        json={
            'prompt': prompt,
            'width': 1024,
            'height': 1024,
        },
    ).json()
    # print(request)
    print(request)
    request_id = request["id"]

    return request_id


def check(id):
    max_tries = 3
    while True:
        time.sleep(0.5)
        result = requests.get(
            'https://api.bfl.ml/v1/get_result',
            headers={
                'accept': 'application/json',
                'x-key': bfl_key,
            },
            params={
                'id': id,
            },
        ).json()
        # print(id)
        if result["status"] == "Ready":
            # print(f"Result: {result['result']['sample']}")
            return result
            break
        else:
            print(f"Status: {result['status']}")
            max_tries-=1
            if max_tries == 0:
                break
            # break

def download_image(url, file_name):
    print(f'Downloading image from {url} for {file_name}')
    response = requests.get(url)
    
    if response.status_code == 200:

        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"Image saved as {file_name}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

def download(names):
    dir_path = './results'
    os.makedirs(dir_path, exist_ok=True)

    jsons = {}
    for name in names:
        with open(f'{dir_path}/{name}.json', 'r') as f:
            str = f.readlines()[0]
            result = check(str.strip())
            jsons[name] = result
        f.close()

    fails = []
    for name in names:

        with open(f'{dir_path}/{name}_result.json', 'w') as f:
            f.write(json.dumps(jsons[name]))
            f.write('\n')
        f.close()

        try:
            download_image(jsons[name]['result']['sample'], f'{dir_path}/{name}.jpg')
        except:
            print(f'Fail in {name}')
            fails.append(name)

    print(fails)

def generate(prompts, names):

    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        result = fluxp_t2i(prompt)
        print(result)
        with open(f'./results/{names[idx]}.json', 'w') as f:
            f.write(result)
            f.write('\n')
            f.write(prompt)
        f.close()

if __name__ == '__main__':

    prompts_all, names_all = get_txt_files(f'../gen_prompts/{sys.argv[1]}')
    for i in range(40):
        prompts = [prompts_all[i]]
        names = [names_all[i]]
        generate(prompts, names)
        download(names)