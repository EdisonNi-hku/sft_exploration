import argparse
import pandas as pd
import random
import pickle

from tqdm import tqdm
from transformers import AutoTokenizer, set_seed, logging
from vllm import LLM, SamplingParams

DEFAULT_SYSTEM = ""


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
    parser.add_argument("--architecture", type=str, default="llama-3")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-*b-hf")
    parser.add_argument("--prompt_file", type=str, default="example_prompts.json")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--max_new_token", type=int, default=512)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--load_tokenizer", action="store_true", default=False)
    parser.add_argument("--logprobs", type=int, default=None)
    return parser.parse_args()


def main(args):
    if args.load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.model_cache_dir, padding_side='left', local_files_only=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir=args.model_cache_dir, padding_side='left', local_files_only=False)

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

    if 'gemma' in args.architecture:
        prompts = [format_prompt(tokenizer, DEFAULT_SYSTEM, p, no_system=True) for p in instructions]
    else:
        prompts = [format_prompt(tokenizer, DEFAULT_SYSTEM, p) for p in instructions]

    if args.sample:
        prompts = random.sample(prompts, args.sample)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        logprobs=args.logprobs,
        max_tokens=args.max_new_token,
    )
    llm = LLM(
        model=args.model_path,
        download_dir=args.model_cache_dir,
        tokenizer=tokenizer,
        dtype='auto',
        seed=args.seed,
        trust_remote_code=args.trust_remote_code
    )

    outputs = llm.generate(prompts, sampling_params)

    generated_text = []
    for output in outputs:
        generated_text.append(output.outputs[0].text)
    output_df = pd.DataFrame({'prompt': prompts, 'output': generated_text})
    output_df.to_csv(args.output_file, index=False)
    with open(args.output_file.replace('.csv', '.pkl'), 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    args = get_args()

    if args.seed >= 0:
        set_seed(args.seed)
        random.seed(args.seed)

    logging.set_verbosity_info()

    main(args)
