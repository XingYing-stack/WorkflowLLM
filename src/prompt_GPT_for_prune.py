import random
import pandas as pd
import pickle
import re
import warnings
from openai import OpenAI
import time
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="./data/synthesized-code_generation_result.json")
parser.add_argument("--save_path", default="./data/pruned_data.json")
parser.add_argument("--s", type=int, required=True)
parser.add_argument("--t", type=int, required=True)
parser.add_argument("--token", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=20)
args = parser.parse_args()

warnings.filterwarnings("ignore")

from pydantic import BaseModel

class thought_and_code(BaseModel):
    thought: str
    code: str

def format_instruction(sample, stat):
    action_names = stat[sample['key']]['action_names']
    type = stat[sample['key']]['type']
    depth = stat[sample['key']]['depth']
    apis_desc = [identifier2python.get(action_name.replace('.', '_')) for action_name in action_names]
    apis_desc = [_ for _ in apis_desc if _ is not None]
    query = sample['query']
    thought = sample['description']
    return [query, apis_desc, thought, type, depth]

def data_clean(old_data, code_col='generated_code_with_comment', api_col='apis'):
    new_data = []
    for sample in old_data:
        if len(sample[code_col]) == 0:
            continue
        code_string = sample[code_col]
        pattern = r'\b(' + '|'.join(list(identifier2python.keys())) + r')\b'
        api_calls = re.findall(pattern, code_string)
        sample[api_col] = set(api_calls) - {'is.workflow.actions.conditional', 'is.workflow.actions.repeat.each', 'is.workflow.actions.repeat.count', 'is.workflow.actions.choosefrommenu',
                                           'is.workflow.actions.gettext', 'is.workflow.actions.dictionary', 'is.workflow.actions.getvalueforkey', 'is.workflow.actions.ask'}
        sample[api_col] = list(sample['apis'])
        new_data.append(sample)
    return new_data

def extract_markdown_dict(input_str):
    match = re.search(r'```json\s*({.*?})\s*```', input_str, re.DOTALL)
    if match:
        try:
            dict_obj = json.loads(match.group(1))
            return dict_obj
        except json.JSONDecodeError:
            return match.group(1)
    else:
        return input_str

new_prune_api = args.token

def prompt_fn(input, model_name):
    client = OpenAI(
        api_key=new_prune_api,
    )
    messages = [{"role": "user", "content": input}]
    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "code_and_thought",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "thought": {"type": "string"}
                },
                "required": ["code", "thought"],
                "additionalProperties": False
            }
        }
    }
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logit_bias={},
        response_format=json_schema
    )
    return completion.choices[0].message.content

def process_sample(ICL_context, query, thought, code, apis):
    final_prompt = f"""
You are exceptionally skilled at polishing tool calling plan (i.e., thought) and python code given a task.

Given task:\n{query}\n\n
Old tool calling plan:\n{thought}\n\n Old code:\n{code}\n\n Used API doc:\n{apis}

Here are examples for you to refer:{ICL_context}.
Please make sure the code is logically correct and operational.

Requirements:
[1] Ensure that both plan and code respond correctly to the task and that code calls match the plan, which you can do by tweaking, embellishing, and modifying both plan and code.
Plan does not have to be one-to-one correspondence of code; plan can be abbreviated.
[2] Please ensure that the code conforms to python syntax. Ensure that all python code is complete and runnable. You can add code when necessary.
[3] Every line of code should be preceded by a comment marked with a “#”. When modifying the code, please modify the in-line comments accordingly.
[4] Ensure that all function parameter calls are correct and you can change the code in case of errors.
[5] Thought and code should be as concise while keeping the meaning intact.
[6] If there are cases including invalid binary code, replace them with reasonable text, delete them, or replace them with a reading operation on a file (especially when the binary code is an encoded image).
Respond strictly with JSON."""
    for _ in range(3):
        try:
            response = prompt_fn(final_prompt, 'gpt-4o-mini')
            response = extract_markdown_dict(response)
            response = json.loads(response)
            thought, code = response["thought"], response["code"]
            return thought, code
        except Exception as e:
            if isinstance(e, json.JSONDecodeError):
                print(f"\nJsonDecodeError with string:\n{response}")
            else:
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
    return None


with open('./data/statistics.pkl', 'rb') as fp:
    stat = pickle.load(fp)


actions_with_freq = Counter([action_name for sample in stat.values() for action_name in sample['action_names']])
num_actions = Counter([len(sample['action_names']) for sample in stat.values()])

with open('./data/identifier2python.pkl', 'rb') as fp:
    identifier2python = pickle.load(fp)

tool_names = set(['_'.join(string.split('_')[:-1]) for string in identifier2python.keys()])
tool_names = {item for item in tool_names if "is_workflow_actions" not in item}
tool_names.add('is_workflow_actions')
tool_names = list(tool_names)
tool_name2api = defaultdict(list)
for tool_name in tool_names:
    for api in identifier2python.keys():
        if api.startswith(tool_name):
            tool_name2api[tool_name].append(api)

with open('./data/sampled_data.json', 'r') as fp:
    data = json.load(fp)
with open('./data/dataset_split_keys.json', 'r') as fp:
    dataset_split = json.load(fp)

with open('./data/statistics.pkl', 'rb') as fp:
    statistics = pickle.load(fp)


with open(args.input_path, 'r') as fp:
    synthesized_data = data_clean(json.load(fp)[args.s:args.t])

data = [sample for sample in data if sample['key'] in stat.keys()]
data = pd.DataFrame(data)
train_keys = set(dataset_split['train'])
data = data[data['key'].isin(train_keys)].to_dict(orient='records')

detailed_data = []
for sample in data:
    detailed_data.append(format_instruction(sample, statistics))

classified_samples = defaultdict(list)
for sample in detailed_data:
    classified_samples[sample[-2]].append(sample)

ICL_number = 1

import concurrent.futures
def multi_thread_wrapper(sample):
    ICL_context = ""
    ICL_type = sample['type']
    ICL_examples = random.sample(classified_samples[ICL_type], ICL_number)
    for _ in range(ICL_number):
        ICL_context += f"""
{{
    "apis": {ICL_examples[_][1]},
    "query": "{ICL_examples[_][0]}",
    "thought": "{ICL_examples[_][2]}"
}}
"""
    query = sample['query']
    thought = sample['generated_thought']
    code = sample['generated_code_with_comment']
    apis = [identifier2python[sample.replace('.', '_')] for sample in sample['apis']]
    apis = '\n\n'.join(apis)
    try:
        pruned_thought, pruned_code = process_sample(ICL_context, query, thought, code, apis)
    except TypeError:
        sample['pruned_thought'] = ""
        sample['pruned_code'] = ""
    else:
        sample['pruned_thought'] = pruned_thought
        sample['pruned_code'] = pruned_code
    return sample

with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    synthesized_data = list(tqdm(executor.map(multi_thread_wrapper, synthesized_data), total=len(synthesized_data)))

try:
    synthesized_data = json.load(open(args.save_path, "r")) + synthesized_data
except Exception:
    pass
with open(args.save_path, 'w') as fp:
    json.dump(synthesized_data, fp, indent=4)
