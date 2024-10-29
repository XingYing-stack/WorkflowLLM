from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pickle
import numpy as np
from transformers import Trainer
import pandas as pd
from datasets import Dataset
import gc
import deepspeed
import torch
from tqdm import tqdm

with open('./data/statistics.pkl', 'rb') as fp:
    stat = pickle.load(fp)
with open('./data/identifier2python.pkl', 'rb') as fp:
    identifier2python = pickle.load(fp)


def clean_cache():
    deepspeed.runtime.utils.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def format_instruction_with_code(sample, add_answer = True, external_data=False):
    if external_data or sample.get('apis', None) is not None:
        action_names = sample['apis']
    else:
        action_names = stat[sample['key']]['action_names']

    python_main_code = sample.get('workflow_code', "")
    thought = sample.get('task_plan', "")
    apis_desc = [identifier2python.get(action_name.replace('.', '_')) for action_name in action_names]
    apis_desc = [_ for _ in apis_desc if _ is not None]
    apis_desc = "\n".join(apis_desc)

    query = sample['query']

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a very helpful AI assistant who can write corresponding Python main code based on user's query and usable Python function interface.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Please generate python main code based on the following query :\n {query}
You can start by using natural language to plan your tool call strategy, and then generate the complete code. For example, `Thought:\n<tool call strategy>\n\nCode:\n```python\n<main code>\n````.
Note that your output should always include `Code:\n```python\n<main code>\n````, formatted accordingly.
Here are some useful function interface you may use:\n {apis_desc}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""
    if add_answer:
        prompt += f"Thought:{thought}\n\nCode:\n```python\n{python_main_code}\n```" + "<|eot_id|>"


    return prompt


def create_loss_mask(tokenizer, input_text, max_length=8192):
    tokenized_inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = tokenized_inputs['input_ids']

    return input_ids



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        clean_cache()

        labels = inputs.get("input_ids")
        loss_mask = inputs.pop("loss_mask", None)

        outputs = model(**inputs)
        # Free unused memory after forward pass
        clean_cache()

        logits = outputs.get("logits")
        logits = logits.float()


        if loss_mask is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().long()
            shift_loss_mask = loss_mask[..., 1:].contiguous().float()  # Apply the same shift to the loss_mask

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size()) * shift_loss_mask
            loss = loss.sum() / (shift_loss_mask.sum() + 1e-5)
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss




class CustomDataCollator:
    def __init__(self, tokenizer, loss_start_token="Thought:", max_length=8192):
        self.tokenizer = tokenizer
        self.loss_start_token = loss_start_token
        self.max_length = max_length

    def __call__(self, features):
        # Extract input_ids from features
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]

        # Find the max sequence length in the batch
        max_len = max(len(ids) for ids in input_ids)

        # Pad input_ids manually to the same length (left padding)
        padded_input_ids = []
        padding_lengthes = []
        for ids in input_ids:
            padding_length = max_len - len(ids)
            padding_lengthes.append(padding_length)
            padded_ids = torch.cat([torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long), ids])
            padded_input_ids.append(padded_ids)
        padded_input_ids = torch.stack(padded_input_ids)

        # Create loss masks and attention masks
        loss_masks = []
        attention_masks = []
        thought_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.loss_start_token)), dtype=torch.long)

        for index, ids in enumerate(padded_input_ids):
            # Find where the "Thought:" token starts
            sliding_windows = ids.unfold(0, thought_token_ids.size(0), 1)
            match_matrix = (sliding_windows == thought_token_ids).all(dim=-1)
            match_indices = torch.nonzero(match_matrix, as_tuple=True)

            # Create a mask initialized to 0
            loss_mask = torch.zeros_like(ids, dtype=torch.float)
            # Initialize attention_mask, 1 for actual tokens, 0 for padding
            attention_mask = torch.ones_like(ids, dtype=torch.float)

            # Padding should be ignored (set to 0 in both loss_mask and attention_mask)
            attention_mask[:padding_lengthes[index]] = 0

            # If the "Thought:" token is found, mask the tokens before its occurrence
            if match_indices[0].numel() > 0:
                thought_start_idx = match_indices[0][0].item()
                loss_mask[thought_start_idx:] = 1

            loss_masks.append(loss_mask)
            attention_masks.append(attention_mask)

        padded_loss_masks = torch.stack(loss_masks)
        padded_attention_masks = torch.stack(attention_masks)

        # Prepare the final batch dictionary
        batch = {
            "input_ids": padded_input_ids,
            "loss_mask": padded_loss_masks,
            "attention_mask": padded_attention_masks  # Added attention_mask to the batch
        }

        return batch


def preprocess_data(data, tokenizer, max_seq_length, format_function):
    input_ids_list = []

    for sample in tqdm(data, desc="tokenizing"):
        input_text = format_function(sample)
        tokenized_inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        input_ids = tokenized_inputs['input_ids']

        input_ids_list.append(input_ids.numpy()[0].tolist())

    df = pd.DataFrame({
        'input_ids': input_ids_list,
    })

    dataset = Dataset.from_pandas(df)
    return dataset



def format_instruction_without_code(sample, add_answer = True):
    text = sample['response']

    if sample.get('query') is not None:
        query = sample.get('query')
    else:
        query = text.split('step_by_step_description')[0].replace("'", "").replace(':', "").replace('{','').replace('}','').replace('query', '').strip()
        if len(query) == 0:
            return None
    if sample.get('task_plan') is not None:
        step_by_step_desc = sample.get('task_plan')
    else:
        step_by_step_desc = text.split('step_by_step_description')[1].replace("'", "").replace(':', "").replace('{','').replace('}','').strip()
        if len(step_by_step_desc) == 0:
            return None
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a very helpful AI assistant who can break down a complicated task into several smaller ones by taking into account the variety of situations and the coherence between the various steps.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Please generate a detailed step-by-step teardown based on the the user's query:\n query: {query}.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

step-by-step teardown:"""
    if add_answer:
        prompt += step_by_step_desc + "<|eot_id|>"

    return prompt


def compute_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    score = sentence_bleu(reference, hypothesis)
    return score


def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure


def print_result(json_file):
    codebleu_scores = []
    ngram_scores = []
    weighted_ngram_scores = []
    syntax_scores = []
    dataflow_scores = []

    for sample in json_file:
        codebleu_scores.append(float(sample['codebleu']))
        ngram_scores.append(float(sample['ngram_match_score']))
        weighted_ngram_scores.append(float(sample['weighted_ngram_match_score']))
        syntax_scores.append(float(sample['syntax_match_score']))
        dataflow_scores.append(float(sample['dataflow_match_score']))

    average_codebleu = np.mean(codebleu_scores)
    average_ngram = np.mean(ngram_scores)
    average_weighted_ngram = np.mean(weighted_ngram_scores)
    average_syntax = np.mean(syntax_scores)
    average_dataflow = np.mean(dataflow_scores)

    print(f"Average CodeBLEU: {average_codebleu}")
    print(f"Average N-gram Match Score: {average_ngram}")
    print(f"Average Weighted N-gram Match Score: {average_weighted_ngram}")
    print(f"Average Syntax Match Score: {average_syntax}")
    print(f"Average Dataflow Match Score: {average_dataflow}")
    return (average_codebleu, average_ngram, average_weighted_ngram, average_syntax, average_dataflow)


def remove_comments(code: str) -> str:
    """
    Remove comments from Python code.

    Args:
    - code (str): The input Python code as a string.

    Returns:
    - str: The Python code with comments removed.
    """
    # Split the input code into lines
    lines = code.split('\n')

    # Initialize an empty list to store code lines without comments
    result = []

    # Iterate over each line
    for line in lines:
        # Find the index of the comment symbol '#'
        comment_index = line.find('#')

        if comment_index != -1:
            # If there is a comment, keep only the part before it
            line = line[:comment_index].rstrip()

        # Add the cleaned line to the result if it's not empty
        if line.strip():
            result.append(line)

    # Join the list back into a single string with newline characters
    return '\n'.join(result)