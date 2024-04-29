import datasets
import argparse
import pandas as pd
import torch
import numpy as np
import random

from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed,\
    logging, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList

from peft import PeftModel


class MultipleTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, token_ids, input_length):
        self.token_ids = set(token_ids)
        self.input_length = input_length

    def __call__(self, input_ids, scores):
        finished = []
        input_ids = input_ids[:, self.input_length:]
        for i in range(input_ids.size()[0]):
            end = False
            for id in self.token_ids:
                if id in input_ids[i]:
                    end = True
                    break
            finished.append(end)

        return all(finished)


SYSTEM = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

DEFAULT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def format_prompt(tokenizer, system, input, no_system=False):
    if no_system:
        chat = [
            {"role": "user", "content": system + '\n\n' + input},
        ]
    else:
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
    formatted_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return formatted_input


def batchify_list(input_list, batch_size):
    # Calculate the number of batches required
    num_batches = (len(input_list) + batch_size - 1) // batch_size

    # Create empty list to hold batches
    batches = []

    # Generate batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(input_list))
        batch = input_list[start_idx:end_idx]
        batches.append(batch)

    return batches


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", type=str, default="cache")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--architecture", type=str, default="llama-2")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-*b-hf")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-*b-hf")
    parser.add_argument("--prompt_file", type=str, default="example_prompts.json")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--max_new_token", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load_peft", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--no_sample", action="store_true", default=False)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--load_8bit", action="store_true", default=False)
    parser.add_argument("--load_tokenizer", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))
    if args.load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.model_cache_dir, trust_remote_code=args.trust_remote_code, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.model_cache_dir, trust_remote_code=args.trust_remote_code, padding_side='left')

    no_original_template = False
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_TEMPLATE
        no_original_template = True
    if args.load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path if not args.load_peft else args.base_model,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=args.model_cache_dir,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )

    else:
        print("Loading 4bit")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path if not args.load_peft else args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.model_cache_dir,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        )
    if args.load_peft:
        print("Resize model to", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_path, local_files_only=True)

    if not args.load_tokenizer:
        if args.architecture == 'llama-1':
            print("Setting EOS, BOS, and UNK tokens for LLama tokenizer")
            tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "<s>",
                    "unk_token": "<unk>",
                }
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if args.architecture in ['llama-2', 'llama-3']:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    if '.' not in args.prompt_file and 'alpaca_eval' in args.prompt_file:
        eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", cache_dir='.')["eval"]
        instructions = [example['instruction'] for example in eval_set]
    else:
        if args.prompt_file.endswith('csv'):
            df = pd.read_csv(args.prompt_file)
        elif args.prompt_file.endswith('xlsx'):
            df = pd.read_excel(args.prompt_file)
        elif args.prompt_file.endswith('json'):
            df = pd.read_json(args.prompt_file)
        else:
            if args.prompt_file.endswith('jsonl'):
                lines = True
            else:
                lines = False
            df = pd.read_json(args.prompt_file, lines=lines)
        instructions = df[args.instruction_field].to_list()

    if 'gemma-it' in args.architecture and not no_original_template:
        prompts = [format_prompt(tokenizer, SYSTEM, p, no_system=True) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<end_of_turn>')]
    elif 'llama-3-it' in args.architecture and not no_original_template:
        prompts = [format_prompt(tokenizer, SYSTEM, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    else:
        prompts = [format_prompt(tokenizer, SYSTEM, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id]
    if args.sample:
        prompts = random.sample(prompts, args.sample)
    prompt_batches = batchify_list(prompts, args.batch_size)

    outputs = []
    transition_scores = []
    for batch in tqdm(prompt_batches, total=len(prompt_batches)):
        input = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=3072,
            return_token_type_ids=False if 'olmo' in args.architecture else None,
        ).to("cuda")
        input_length = input.input_ids.shape[1]
        output = model.generate(
            **input,
            stopping_criteria=StoppingCriteriaList([MultipleTokenStoppingCriteria(eos_token_ids, input_length)]),
            generation_config=GenerationConfig(
                do_sample=not args.no_sample,
                max_new_tokens=args.max_new_token,
                top_p=1 if args.no_sample else args.top_p,
                temperature=1 if args.no_sample else args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        )
        output_texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=False)
        outputs.extend(output_texts)

        # transition_score = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        # generated_tokens = output.sequences[:, input_length:]
        # for i in range(len(transition_score)):
        #     transition_score_pairs = []
        #     for tok, score in zip(generated_tokens[i], transition_score[i]):
        #         transition_score_pairs.append((tokenizer.decode(tok), np.exp(score.cpu().numpy())))
        #     transition_scores.append(transition_score_pairs)

    # import pickle
    # with open(args.output_file.replace('.csv', '.pkl'), 'wb') as f:
    #     pickle.dump(transition_scores, f)
    output_df = pd.DataFrame({'output': outputs})
    output_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = get_args()

    if args.seed >= 0:
        set_seed(args.seed)

    logging.set_verbosity_info()

    main(args)

