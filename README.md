# sft_exploration
Training and evaluation pipelines for SFT exploration.

### How to run SFT?
There are two settings, the first using huggingface SFTTrainer, which support packing, and training on completion only. 
**However, for models like Llama-2 and Mistral-7b without padding tokens, SFTTrainer will not learn the loss from EOS token, thus resulting in endless generation.**
But models with a padding token would be fine with SFTTrainer. 
```shell
# First config run_sft_qlora.sh
cd code
bash run_sft_qlora.sh google/gemma-7b flan_sampled_100k_zephyr.json HUGGINGFACE_CACHE g7b_flan_100k_zephyr False True
```
- The above command will run QLoRA fine-tuning on google/gemma-7b with a subset of 100k flan data (in a form of {"input": ..., "output": ...}).
- The trained adapters will be saved in g7b_flan_100k_zephyr
- The first "False" means no packing. The second "True" means do training on source as well.

Another script use Seq2SeqTrainer with a special data collator, which always ensures EOS tokens are learned. However, packing is not supported here.
```shell
cd code
bash run_original_qlora.sh google/gemma-7b flan_sampled_100k_zephyr.json HUGGINGFACE_CACHE g7b_flan_100k_zephyr True
```

### How to run DPO?
Our DPO part is adapted from huggingface alignment-handbook.
```shell
cd code
accelerate launch --config_file ./multi_gpu.yaml --num_processes=1 ./run_dpo.py ./config_qlora.yaml
```
This command will start a DPO fine-tuning on argilla/dpo-mix-7k. Remember to config config_qlora.yaml properly.
