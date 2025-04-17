from openai import OpenAI
import os
import json
import time
from datetime import datetime, timedelta

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def gpt4o(prompt):
    client = OpenAI(api_key='YourAPIKey')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Do not add any greetings."},
            {"role": "user", "content": prompt}
        ],
        temperature= 0.9,

    )
    return response

# CC IEI
def interp2topic(datum):
    interpretation = datum['interpretation']
    prompt = f"【{interpretation}】\nGive me the theme using 1 or 2 words. \nProvide the most important visual objects for me.\nAnswer in this form:\n Topic:[the topic of the interpretation]\nObject:[enumerate the objects, split with comma]"
    result = gpt4o(prompt)
    return result

# Baseline II
def topic2prompt_baseline(datum, topic):
    prompt = f"""help me to create an image of minimalistic style on the topic of 【{topic}】.\nCombine the below objects in the image:\n {datum['primary']}\nGive me the image generation prompt."""
    result = gpt4o(prompt)
    return result

def topic2prompt_opno(datum, topic):
    prompt = f"""Help me to create an image of minimalistic style on the topic of 【{topic}. 
Only include the below objects in the image:
【{datum['primary']}】
Here are the creation steps:
1) You want to create a mashup image that combines the objects into one object. 
2) It is better for you to think about their visual similarity such as shapes, size, and color. 
3) And you want to replace part of the one object with another's.
4) Make sure the whole image contains only one object rather than multiple objects.
Give me the image generation prompt."""

    result = gpt4o(prompt)
    return result

def topic2prompt_noo(datum, topic):
    prompt = f"""help me to create an image of minimalistic style on the topic of 【{topic}】.
The creation should follow the formula: 【Object A】 + 【Object B】 -> 【Object A】 + 【Object C】.
Object A is {datum['obja']}, Object B is {datum['combination'][0]['original']} , Object C is {datum['combination'][0]['new']}.
Here are the creation steps:
1) Think about the commonly display of {datum['obja']} and {datum['combination'][0]['original']} via functionality or common usage. 
2) Fully Replace {datum['combination'][0]['original']} to {datum['combination'][0]['new']} and create a novel and meaningful image to express the theme. Don't keep any part of {datum['combination'][0]['original']}.
3) Double check that the image should only consist of {datum['obja']} and {datum['combination'][0]['new']}.
Please only give me the final image generation prompt. """
    result = gpt4o(prompt)
    return result

def topic2prompt_no(datum, topic):
    prompt = f"""help me create an image of minimalistic style on the topic of 【{topic}】
The creation should follow the formula: 【Object A】 -> 【Object B】
Object A is {datum['combination'][0]['original']} and , Object B is {datum['combination'][0]['new']}
Here are the creation steps:
1) You want to think about their visual similarity such as shapes. 
3) And you want to replace the 【Object A】 totally with the 【Object B】and create a novel and meaningful image to express the theme.
4) Make sure the whole image contains only one object rather than multiple objects.
Give me the image generation prompt."""
    result = gpt4o(prompt)
    return result

def topic2prompt_d(datum, topic):
    prompt = f"""help me create an image of minimalistic style on the topic of 【{topic}】
The creation should follow the formula: 【Object A】+【Object B】
Object A and Object B are in {datum['primary']}
Here are the creation steps:
1) From the message of 【{topic}】, you may figure out the relationship between the two objects. Brainstorm as many ideas as possible.
2) Choose the most suitable one and combine the objects together. 
3) Double check whether the whole image expresses【{topic}】and contains only the objects in the list. 
Only give me the final image generation prompt."""
    result = gpt4o(prompt)
    return result


def load_data():
    path = './clean_data.json'
    with open(f'{path}', 'r') as file:
        raw_data = json.load(file)

    return raw_data


def beijing_time(current_utc_time):
    # current_utc_time = time.time()
    utc_time = datetime.utcfromtimestamp(current_utc_time)
    beijing_time = utc_time + timedelta(hours=8)
    formatted_beijing_time = beijing_time.strftime('%Y%m%d_%H%M%S')
    return formatted_beijing_time


def process_datum(datum):

    result = interp2topic(datum)
    content = result.choices[0].message.content
    topic = content.split('\n')[0]
    current_time = beijing_time(time.time())

    prompt_base = topic2prompt_baseline(datum, topic)
    baseline_filename = f'./gen_prompts/baseline/_{datum["id"]}_baseline_{current_time}.txt'
    with open(baseline_filename, 'w') as f:
        f.write(prompt_base.choices[0].message.content)
        f.write('\n')
        f.write(str(prompt_base))

    if datum['type'] == 'A':
        prompt_cc = topic2prompt_opno(datum, topic)
    elif datum['type'] == 'B':
        prompt_cc = topic2prompt_no(datum, topic)
    elif datum['type'] == 'C':
        prompt_cc = topic2prompt_noo(datum, topic)
    elif datum['type'] == 'D':
        prompt_cc = topic2prompt_d(datum, topic)

    cc_filename = f'./gen_prompts/cc/_{datum["id"]}_cc_{current_time}.txt'
    with open(cc_filename, 'w') as f:
        f.write(prompt_cc.choices[0].message.content)
        f.write('\n')
        f.write(str(prompt_cc))

    
if __name__ == '__main__':

    data = load_data()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_datum, datum) for datum in data]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass