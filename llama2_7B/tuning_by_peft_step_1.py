import os
import sys

import numpy as np
import torch.nn as nn
import torch
# import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    AdamW,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, PeftModel, PeftConfig
from trl import SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ********************************  ********************************

class MLP(nn.Module):
    """

    """
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.activation = nn.ReLU()

        # He
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)  #
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x


def count_non_empty_lines(file_path):
    """

    :param file_path:
    :return:
    """
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  #
                count += 1
    return count


# ********************************  ********************************

#
model_name = "llama2_7B_chat_hf/"
#
dataset_train_path = "dataset/build_data/dataset/contrastive/train.txt"
dataset_eval_path = "dataset/build_data/dataset/contrastive/test_100.0.txt"
#
new_model = "./new_model/llama-2-7b_chat_contrast_based"
################################################################################
#
################################################################################
lora_r = 64  # LoRA attention dimension
lora_alpha = 16  # Alpha parameter for LoRA scaling
lora_dropout = 0.1  # Dropout probability for LoRA layers

################################################################################
# bitsandbytes
################################################################################
use_4bit = True  # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
use_nested_quant = False  # Activate nested quantization for 4-bit base models (double quantization)

################################################################################
#
################################################################################
output_dir = "temp_result"  # Output directory where the model predictions and checkpoints will be stored
num_train_epochs = 1  # Number of training epochs

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True  #

per_device_train_batch_size = 512  # Batch size per GPU for training
per_device_eval_batch_size = 512  # Batch size per GPU for evaluation
gradient_accumulation_steps = 1  # Number of update steps to accumulate the gradients for
gradient_checkpointing = True  # Enable gradient checkpointing
max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)
learning_rate = 5e-6  # Initial learning rate (AdamW optimizer)
weight_decay = 0.001  # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"  # Optimizer to use
lr_scheduler_type = "cosine"  # Learning rate schedule (constant a bit better than cosine)
max_steps = -1  # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False
save_steps = 500  # Save checkpoint every X updates steps
logging_steps = 50  # Log every X updates steps
evaluation_strategy = "no"
################################################################################
# SFT
################################################################################
max_seq_length = 30  # Maximum sequence length to use
packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0}  # Load the entire model on the GPU 0

print("=" * 80)
print(f"""
The main parameters of the current model are set as follows:
    Batch Size: {per_device_train_batch_size}
    Learning Rate: {learning_rate}
    Optim: {optim}
    LR Scheduler: {lr_scheduler_type}
    max_seq_length:{max_seq_length}
    Warmup Ratio:{warmup_ratio}
    Epoch:{num_train_epochs}
    Train Count: {count_non_empty_lines(dataset_train_path)}
    Eval Count: {count_non_empty_lines(dataset_eval_path)}
    Modal Name:{new_model}
""")
print("=" * 80)

################################################################################
#
################################################################################
# ******************  ******************
dataset = load_dataset("text", data_files={
    "train": dataset_train_path,
    "test": dataset_eval_path
})

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
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    use_flash_attention_2=True,
    output_hidden_states=True
)


model.config.use_cache = False
model.config.pretraining_tp = 1

"""

"""
# optimizer
mlp = MLP().to("cuda")

model_parameters = [param for name, param in model.named_parameters() if param.requires_grad]


mlp_parameters = [param for name, param in mlp.named_parameters() if param.requires_grad]


combined_parameters = model_parameters + mlp_parameters
optimizer = AdamW(combined_parameters, lr=learning_rate, weight_decay=weight_decay)


total_steps = count_non_empty_lines(dataset_train_path) // (
        per_device_train_batch_size * gradient_accumulation_steps) * num_train_epochs


lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_ratio * total_steps,
    num_training_steps=total_steps
)
optimizers = (optimizer, lr_scheduler)

"""

"""

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("llama2_7B_chat_hf/", trust_remote_code=True)
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


"""

"""


# ******************  ******************
class ContrasTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)


        sequence_output = (outputs.hidden_states[-1] + outputs.hidden_states[0]) / 2


        batch_emb = sequence_output.mean(dim=1)

        tensor_np = batch_emb.detach().cpu().numpy()



        batch_emb = mlp(batch_emb)

        # simcse_loss
        batch_size = batch_emb.size(0)

        y_true = torch.cat([
            torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
            torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)
        ], dim=1).view(-1).to("cuda")

        norm_emb = F.normalize(batch_emb, dim=1, p=2)

        sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))


        sim_score = sim_score - torch.eye(batch_size).to(sim_score.device) * 1e12


        sim_score = sim_score * 20
        loss = torch.nn.CrossEntropyLoss()(sim_score, y_true)

        return (loss, outputs) if return_outputs else loss


# ****************** Trainer ******************

trainer = ContrasTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    optimizers=optimizers
)
# ****************** Trainer ******************


################################################################################
#  Train
################################################################################
# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)
