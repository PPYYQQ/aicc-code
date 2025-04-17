import json
from vllm import LLM, SamplingParams

def LoadModel(model_path):
    llm = LLM(
            model=model_path,
            gpu_memory_utilization = 0.8,
            tensor_parallel_size = 4
        )
    
    sampling_params = SamplingParams(
        top_k = 10,
        top_p = 0.95,
        temperature = 0.5,
        max_tokens = 2048,
        frequency_penalty = 1.2,
        stop_token_ids=[128009]
    )
    return llm, sampling_params

def LoadPrompts(original_object):
    with open("./prompt_template.txt", 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    file.close()
    
    with open(f"./generic/{original_object}.json", 'r', encoding='utf-8') as file:
        object_feature = json.load(file)
    file.close()

    image_topic_prompts = []
    image_prompts = []
    for feature in object_feature.keys():
        object1 = original_object    
        for object2 in object_feature[feature]:
            prompt = prompt_template.replace('object1', object1).replace('object2', object2)
                    
            image_topic_prompts.append((original_object, feature, prompt))
            image_prompts.append(prompt)
    return image_topic_prompts, image_prompts

def main():

    # Load Model
    model_path = "/home/yongqian/.cache/huggingface/hub/models--nvidia--Llama-3.1-Nemotron-70B-Instruct-HF/snapshots/fac73d3507320ec1258620423469b4b38f88df6e"
    llm, sampling_params = LoadModel(model_path)

    # Load propmts
    original_object = 'perfume'
    image_topic_prompts, image_prompts = LoadPrompts(original_object)
   
    # Generate prompts for t2i models
    outputs = llm.generate(image_prompts, sampling_params)

    pure_data = []

    with open(f'image_prompts_from1_{original_object.replace(" ", "_")}.txt', 'w', encoding='utf-8') as f:
        for idx, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            f.write(f"----------divider----------\n")
            f.write(f"Wide Topic: {image_topic_prompts[idx][0]}\n")
            f.write(f"Topic: {image_topic_prompts[idx][1]}\n")
            f.write(f"Prompt: {prompt!r}, Generated text: {generated_text!r}\n")
            f.write(f"----------divider----------\n")
            temp_data = {
                "Wide Topic": f"{image_topic_prompts[idx][0]}",
                "Topic": f"{image_topic_prompts[idx][1]}",
                "inputPrompt": f"{prompt!r}", 
                "output": f"{generated_text!r}"
            }
            pure_data.append(temp_data)

    with open(f'prompts_from1_{original_object.replace(" ", "_")}.json', 'w', encoding='utf-8') as file:
        json.dump(pure_data, file, ensure_ascii=False, indent=4)

    print("Data has been written to output.json")

if __name__ == "__main__":
    main()


