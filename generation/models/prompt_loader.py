import os

def get_txt_files(directory):
    txt_files_paths = []
    txt_files_names = []
    prompts = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                txt_files_paths.append(file_path)
                txt_files_names.append(file.replace('.txt',''))


    for txt_files_path in txt_files_paths:
        with open(txt_files_path, 'r') as f:
            prompt = f.readlines()[0]
        prompts.append(prompt)

    
    return prompts, txt_files_names


directory = './prompts'
prompts, names = get_txt_files(directory)

# print("Names:", names)

print(prompts[0])