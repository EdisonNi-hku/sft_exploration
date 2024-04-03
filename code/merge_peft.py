from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    model_cache_dir: Optional[str] = field(default='./cache', metadata={"help": "Path for huggingface cache"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
    assert script_args.base_model_name is not None, "please provide the name of the Base model"
    assert script_args.output_name is not None, "please provide the output name of the merged model"

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=script_args.model_cache_dir,
        local_files_only=True,
    )

    model = PeftModel.from_pretrained(base_model, script_args.adapter_model_name, local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(script_args.adapter_model_name, local_files_only=False)

    model.merge_and_unload()

    model.save_pretrained(f"{script_args.output_name}")
    tokenizer.save_pretrained(f"{script_args.output_name}")


if __name__ == '__main__':
    main()






