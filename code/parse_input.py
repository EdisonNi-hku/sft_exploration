import json
import pandas as pd
import argparse
import random
from transformers import AutoTokenizer

SYSTEM = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""


def format_instruction(tokenizer, system, input, output, no_system=False, only_generation=True):
    if no_system:
        chat_full = [
            {"role": "user", "content": system + '\n\n' + input},
            {"role": "assistant", "content": output},
        ]
        chat_prompt = [
            {"role": "user", "content": system + '\n\n' + input},
        ]
    else:
        chat_full = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
            {"role": "assistant", "content": output},
        ]
        chat_prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
    formatted_input = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=only_generation)
    formatted_full = tokenizer.apply_chat_template(chat_full, tokenize=False, add_generation_prompt=False)
    formatted_output = formatted_full.replace(formatted_input, '')
    return formatted_input, formatted_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--cache", type=str, default=".")
    parser.add_argument("--input_field", type=str, default="instruction")
    parser.add_argument("--output_field", type=str, default="response")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--only_generation", action='store_true', default=False)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache, local_files_only=False)

    # tokenizer = AutoTokenizer.from_pretrained(args.arch)

    if args.data_path.endswith('json'):
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        instructions = [d[args.input_field] for d in data]
        responses = [d[args.output_field] for d in data]
    else:
        df = pd.read_csv(args.data_path)
        df.dropna(subset=[args.output_field, args.input_field], inplace=True)
        instructions = df[args.input_field].to_list()
        responses = df[args.output_field].to_list()

    parsed = []
    is_gemma = True if 'gemma' in args.model else False
    for i, r in zip(instructions, responses):
        parsed_input, parsed_output = format_instruction(tokenizer, SYSTEM, i, r, no_system=is_gemma, only_generation=args.only_generation)
        parsed.append({"input": parsed_input, "output": parsed_output})

    if args.sample > 0:
        random.seed(42)
        parsed = random.sample(parsed, args.sample)

    print(len(parsed))
    with open(args.output_path, 'w') as f:
        json.dump(parsed, f)
