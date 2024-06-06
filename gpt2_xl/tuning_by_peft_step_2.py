import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer


def count_non_empty_lines(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                count += 1
    return count


# ********************************  ********************************


model_name = "gpt2-xl/"
dataset_train_path = "dataset/build_data/dataset/04/v3_train_10k.txt"
dataset_eval_path = "dataset/build_data/dataset/04/v3_test_1.6k.txt"
new_model = "./new_model/gpt2-xl_contrast_10k_8_1_1data_3epoch"
################################################################################
# LoRA
################################################################################
lora_r = 64  # LoRA attention dimension
lora_alpha = 16  # Alpha parameter for LoRA scaling
lora_dropout = 0.1  # Dropout probability for LoRA layers

################################################################################
# bitsandbytes 4bit
################################################################################
use_4bit = True  # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
use_nested_quant = False  # Activate nested quantization for 4-bit base models (double quantization)

################################################################################

################################################################################
output_dir = "./results"  # Output directory where the model predictions and checkpoints will be stored
num_train_epochs = 3  # Number of training epochs

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True  #

per_device_train_batch_size = 32  # Batch size per GPU for training
per_device_eval_batch_size = 32  # Batch size per GPU for evaluation
gradient_accumulation_steps = 1  # Number of update steps to accumulate the gradients for   梯
gradient_checkpointing = True  # Enable gradient checkpointing
max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)
learning_rate = 1e-4  # Initial learning rate (AdamW optimizer)
weight_decay = 0.001  # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"  # Optimizer to use
lr_scheduler_type = "cosine"  # Learning rate schedule (constant a bit better than cosine)
max_steps = -1  # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)

print("#" * 80)
print(f"""
The main parameters of the current model are set as follows:
Batch Size: {per_device_train_batch_size}
Learning Rate: {learning_rate}
Optim: {optim}
LR Scheduler: {lr_scheduler_type}
Warmup Ratio:{warmup_ratio}
Epoch:{num_train_epochs}
Train Count: {count_non_empty_lines(dataset_train_path)}
Eval Count: {count_non_empty_lines(dataset_eval_path)}
Modal Name:{new_model}
""")
print("#" * 80)

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
save_steps = 125  # Save checkpoint every X updates steps
logging_steps = 50  # Log every X updates steps
evaluation_strategy = "steps"  #
################################################################################
# SFT
################################################################################
max_seq_length = 512  # Maximum sequence length to use
packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0}  # Load the entire model on the GPU 0

################################################################################
#
################################################################################
# ****************** ******************
dataset = load_dataset("text", data_files={
    "train": dataset_train_path,
    "test": dataset_eval_path
})

dataset = dataset.shuffle(seed=42)

# dataset = load_dataset(dataset_name, split="train")
# ******************  ******************

# ******************  ******************
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# ******************  ******************


# ******************  Model Tokenizer******************

model = AutoModelForCausalLM.from_pretrained(
    # contrastive pretrain
    "new_model/gpt2_xl_contrast_based_merge_lora",
    # model_name,   # without contrastive pretrain
    quantization_config=bnb_config,
    device_map=device_map,
    # use_flash_attention_2=True
    # attn_implementation="flash_attention_2"
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ******************  Model Tokenizer******************

# ****************** LoRA ******************
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
# ****************** LoRA ******************
# model = PeftModel.from_pretrained(model, new_model)
# model = model.merge_and_unload()

# ******************  ******************
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy=evaluation_strategy,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb"
    # report_to="tensorboard"
)
# ******************  ******************

# ****************** Trainer ******************
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing
)
# ****************** Trainer ******************


################################################################################
#  Train
################################################################################
# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
