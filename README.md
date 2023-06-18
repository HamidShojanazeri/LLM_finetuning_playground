# LLM_finetuning_playground

To run with FSDP and PEFT methods use the following command

```bash
pip install -r requirements.txt
torchrun --nnodes 1 --nproc_per_node 4  fsdp_finetuning.py fsdp_finetuning.py --enable_fsdp --use_peft --peft_method lora
```