from utils_for_llm import *
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainerState, TrainerControl
from datasets import DatasetDict, Dataset
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
import re
import argparse
import warnings
from accelerate import Accelerator
from accelerate.utils import gather_object
from codebleu import calc_codebleu
import os
import torch.distributed as dist
from datetime import timedelta
import time
from transformers import DataCollatorWithPadding

# Ignore all warnings
warnings.filterwarnings("ignore")


os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = "False"

if os.getenv('PYCHARM_HOSTED') != '1':
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=6))


# Initialize the Accelerator
accelerator = Accelerator(mixed_precision='bf16')

if accelerator.state.deepspeed_plugin:
    deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
    zero_version = deepspeed_config.get('zero_optimization', {}).get("stage")
    print(zero_version)
else:
    zero_version = -1

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", default='./data/seed_data.json', type=str) # or ./data/synthesized_data.json
parser.add_argument("--save_strategy", default="epoch", type=str)
parser.add_argument("--output_dir_name", default=None, type=str)
parser.add_argument("--model_id", default="Meta-Llama-3.1-8B-Instruct", type=str)
parser.add_argument("--do_train",action="store_true")
parser.add_argument("--do_infer",action="store_true")
parser.add_argument("--OOD", action="store_true")
parser.add_argument("--load_path", default="", type=str)
parser.add_argument("--neftune_noise_alpha", default=None, type=float)
parser.add_argument("--debug",action="store_true")
parser.add_argument("--eval_steps", default=500, type=float)
parser.add_argument("--epochs", default=5, type=int)



args = parser.parse_args()
task = "code_generation"
model_id = args.model_id


# =============format_instruction===============
max_seq_length = 8192
if task == "code_generation":
    format_instruction = format_instruction_with_code
    BATCH_SIZE = 1
    target_col = "workflow_code"
elif task == "task_breakdown":
    format_instruction = format_instruction_without_code
    max_seq_length = 768
    BATCH_SIZE = 1
    target_col = "task_plan"

else:
    raise Exception(f'{task} is not defined.')
eval_batch_size = 2 * BATCH_SIZE
# =================================================




# Define prediction function using accelerate with distributed processing
def predict_on_validation_BATCH(model, tokenizer, eval_dataset, batch_size=12, external_data=False):
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_inputs = [format_instruction(sample, add_answer=False, external_data=external_data) for sample in eval_dataset]
    tokenized_inputs = [tokenizer(sample, max_length=max_seq_length, padding=True, truncation=True, add_special_tokens=False)
                        for sample in eval_inputs]

    eval_loader = torch.utils.data.DataLoader(tokenized_inputs, batch_size=batch_size, collate_fn=collator, shuffle=False, drop_last=False)

    total_index = 0
    ans = []
    # Initialize progress bar
    progress_bar = tqdm(total=len(eval_loader), desc=f"Process {accelerator.process_index}", leave=False,
                        disable=not accelerator.is_local_main_process)

    for batch in eval_loader:
        with torch.inference_mode():
            inputs = {key: value.to(accelerator.device) for key, value in batch.items() if key != 'labels'}
            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=tokenizer.pad_token_id,
                                     max_length=max_seq_length,  num_return_sequences=1)

        original_lengths = [len(input_ids) for input_ids in inputs['input_ids']]

        for i, original_length in enumerate(original_lengths):
            response = tokenizer.decode(outputs[i][original_length:], skip_special_tokens=True)

            # RE match "Thought" part
            pattern = r"Thought:([\s\S]*?)(?:Code:|$)"
            match = re.search(pattern, response)

            if match:
                generated_thought = match.group(1).strip()
            else:
                generated_thought = ""
            # extract ```python ```
            generated_code_lst = re.findall(r"```python(.*?)(```|$)", response, re.DOTALL)
            generated_code_lst = [sample[0] for sample in generated_code_lst]
            if len(generated_code_lst):
                generated_code_without_comment = remove_comments(generated_code_lst[0])
                generated_code_with_comment = generated_code_lst[0]
            else:
                generated_code_without_comment = ""
                generated_code_with_comment = ""
            # remove comment
            if target_col in eval_dataset[total_index]:
                reference_code_without_comment = remove_comments(eval_dataset[total_index][target_col])
                reference_code_with_comment = eval_dataset[total_index][target_col]
            else:
                reference_code_without_comment = ""
                reference_code_with_comment = ""
            if 'task_plan' in eval_dataset[total_index]:
                gold_thought = eval_dataset[total_index]['task_plan']
            else:
                gold_thought = eval_dataset[total_index].get('thought', "")
            query = tokenizer.decode(outputs[i][:original_length], skip_special_tokens=True)
            sample_result = {
                'query': query,
                'gold_code_with_comment': reference_code_with_comment,
                'gold_code_without_comment': reference_code_without_comment,
                'gold_thought' : gold_thought,
                'generated_code_with_comment' : generated_code_with_comment,
                'generated_code_without_comment': generated_code_without_comment,
                'generated_thought': generated_thought,
                'category': eval_dataset[total_index].get('category', "NA"),
                'type': eval_dataset[total_index].get('type', "NA"),
                'apis': eval_dataset[total_index].get('apis', [])
            }
            eval_result = calc_codebleu([reference_code_without_comment], [generated_code_without_comment], lang="python", weights=(0.1, 0.1, 0.4, 0.4), tokenizer=None)
            sample_result.update(eval_result)
            total_index += 1
            ans.append(sample_result)
        torch.cuda.empty_cache()
        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()

    model.train()
    return ans


class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, output_path, step_interval=20):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.step_interval = step_interval

        self.best_codebleu = -1  # Initialize best CodeBLEU score
        self.best_model_path = os.path.join(output_path, 'best')  # Initialize path to best model

        # Make sure the output directory exists
        os.makedirs(self.best_model_path, exist_ok=True)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.step_interval == 0:
            self.evaluate(args, state, control)

    def evaluate(self, args, state: TrainerState, control: TrainerControl):
        # sync GPUs and start the timer
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        start = time.time()
        # Split the data across processes
        with accelerator.split_between_processes(self.eval_dataset) as eval_dataset:
            infer_result = predict_on_validation_BATCH(model, self.tokenizer, eval_dataset, batch_size=eval_batch_size)

        # Gather results from all processes
        infer_result = gather_object(infer_result)
        timediff = time.time() - start

        minutes, seconds = divmod(timediff, 60)
        hours, minutes = divmod(minutes, 60)

        # Only save the results on the main process
        if accelerator.is_main_process:
            print(f"Inference Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
            with open(os.path.join('./output', f'{path}_step{state.global_step}_result.json'), 'w') as fp:
                json.dump(infer_result, fp, indent=4)
            print('Result is dumped to:\n ',os.path.join('./output', f'{path}_step{state.global_step}_result.json'))
            metrics = print_result(infer_result)
            print('=' * 50)

            codebleu = metrics[0]

            # Save model if CodeBLEU is better
            if codebleu > self.best_codebleu:
                self.best_codebleu = codebleu
                print(f"New best CodeBLEU: {self.best_codebleu:.4f}, saving model to {self.best_model_path}")

                trainer.save_model(output_dir=self.best_model_path)


        torch.cuda.empty_cache()



if __name__ == "__main__":
    model_path = args.load_path if args.load_path else model_id
    last_part = os.path.basename(model_path)
    if last_part.lower() == 'best':
        second_last_part = os.path.basename(os.path.dirname(model_path))
        last_part = os.path.join(second_last_part, last_part).replace('/', '_')

    path = last_part
    if args.neftune_noise_alpha:
        path += f"-neft{args.neftune_noise_alpha}"
    if args.OOD:
        path += '_OOD'

    file_name = os.path.splitext(os.path.basename(args.train_file))[0]
    path += f"_{file_name}"

    data = pd.read_json(args.train_file).to_dict(orient='records')
    if args.OOD == False:
        with open('./data/dataset_split_keys.json', 'r') as fp:
            dataset_split = json.load(fp)
    else:
        with open('./data/dataset_split_keys_ood.json', 'r') as fp:
            dataset_split = json.load(fp)
    data = [sample for sample in data if (sample['key'] in stat.keys() or sample['key'] in {'synthesized_training_data', 'synthesized_ood_test_data'})] # NOTE: Here we add the key for the synthesized data, which must be synthesized_data.
    # ===================================

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # ==================================

    data = pd.DataFrame(data)

    train_keys = set(dataset_split['train']) | {'synthesized_training_data'}
    dev_keys = set(dataset_split['dev'])

    train_df = data[data['key'].isin(train_keys)]
    val_df = data[data['key'].isin(dev_keys)]

    # =======debug======
    if args.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(10)

        args.eval_steps = 2
        max_seq_length = 8192
        path = 'DEBUG_' + path

    # =======debug======

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    dataset_dict = DatasetDict({
        'train': preprocess_data(train_dataset, tokenizer, max_seq_length, format_function=format_instruction),
        'validation': val_dataset,
    })

    # ============PRINT PARAMS============
    if accelerator.is_main_process:
        print('path:', path)
        print('epoch:', args.epochs)
        print('OOD:', args.OOD)
        print(f'batch_size:{BATCH_SIZE} eval_batch_size:{eval_batch_size}')
        print(f'max_seq_length:{max_seq_length}')
        print(f'Load from: {model_path}')
        print('Training Length', train_df.shape[0], 'DEV Length', val_df.shape[0])
    # =================================

    use_flash_attention = True
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if zero_version != 3:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2",
            device_map={"": accelerator.process_index},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
    model.config.pretraining_tp = 1

    if args.do_infer and zero_version != 3:
        # =============Infer before Training==============
        # sync GPUs and start the timer
        accelerator.wait_for_everyone()
        start = time.time()
        # Split the data across processes
        with accelerator.split_between_processes(dataset_dict['validation']) as eval_dataset:
            infer_result = predict_on_validation_BATCH(model, tokenizer, eval_dataset, batch_size=eval_batch_size)

        # Gather results from all processes
        infer_result = gather_object(infer_result)
        timediff = time.time() - start

        minutes, seconds = divmod(timediff, 60)
        hours, minutes = divmod(minutes, 60)


        # Only save the results on the main process
        if accelerator.is_main_process:
            print('=' * 25, 'DEV','='* 25)
            print(f"Inference Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
            with open(os.path.join('./output', f'{path}_init_DEV_result.json'), 'w') as fp:
                json.dump(infer_result, fp, indent=4)
            print('Init reuslt is dumped to:\n ', os.path.join('./output', f'{path}_init_DEV_result.json'))
            print_result(infer_result)
            print('=' * 50)
        # =============Infer before training=============

    if args.do_train:
        if zero_version == 3 or args.save_strategy == 'epoch':
            callbacks = None
            save_strategy='epoch'
        else:
            callbacks = [EvaluationCallback(dataset_dict['validation'], tokenizer, path, step_interval=args.eval_steps)]
            save_strategy='no'


        training_args = SFTConfig(
            output_dir=path,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy=save_strategy,
            learning_rate=2e-5,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            disable_tqdm=False,
            report_to="tensorboard",
            neftune_noise_alpha = args.neftune_noise_alpha,
            max_seq_length=max_seq_length
        )

        loss_start_token = "<|start_header_id|>assistant<|end_header_id|>"
        data_collator = CustomDataCollator(tokenizer=tokenizer, loss_start_token=loss_start_token, max_length=max_seq_length)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict['train'],
            tokenizer=tokenizer,
            callbacks=callbacks,
            data_collator=data_collator  # dynamic padding
        )

        trainer.train()

        clean_cache()
        if accelerator.is_main_process:
            print('Training is over.')
            print('Start to dump!')

        if args.output_dir_name:
            output_path = os.path.join(path, args.output_dir_name)
        else:
            output_path = None
        trainer.save_model(output_path)

        if accelerator.is_main_process:
            print('Dump over.')


        # =============Infer After Training==============
        del trainer
        clean_cache()
        # sync GPUs and start the timer
        accelerator.wait_for_everyone()
        start = time.time()
        # Split the data across processes
        with accelerator.split_between_processes(dataset_dict['validation']) as eval_dataset:
            infer_result = predict_on_validation_BATCH(model, tokenizer, eval_dataset, batch_size=eval_batch_size)

        # Gather results from all processes
        infer_result = gather_object(infer_result)
        timediff = time.time() - start

        minutes, seconds = divmod(timediff, 60)
        hours, minutes = divmod(minutes, 60)


        # Only save the results on the main process
        if accelerator.is_main_process:
            print('=' * 25, 'DEV','='* 25)
            print(f"Inference Time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
            with open(os.path.join('./output', f'{path}_DEV_final.json'), 'w') as fp:
                json.dump(infer_result, fp, indent=4)
            print('Final reuslt is dumped to:\n ', os.path.join('./output', f'{path}_DEV_final.json'))
            print_result(infer_result)
            print('=' * 50)