import json
import sys

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

################################################################################
# tokenizer
################################################################################
#
model_name = "mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

################################################################################
#
################################################################################
print("************************* Hyperparameters *************************")
COUNT = 20
dataset_path = f"dataset/build_data/generate_dataset/generate_{COUNT}.json"
new_model_base_path = "mistral/new_model/"
# peft_model_id = "mistral-7b_chat_10k_8_1_1data_3epoch_v2"
peft_model_id = "mistral-7b_chat_contrast_10k_8_1_1data_3epoch_v2"
new_peft_model = new_model_base_path + peft_model_id

num_sentence = 10
batch_size = 1
max_new_token = 50
diversity_penalty = 1.0
device = "cuda:0"
device_map = {"": 0}

save_data_path = f"./new_datas/{peft_model_id}_new_data_{COUNT}_sentence_{num_sentence}_penalty_{diversity_penalty}.json"

print(
    f"""
    model name：{peft_model_id}
    batch size: {batch_size}
    diversity penalty：{diversity_penalty}
    """
)

################################################################################
# bitsandbytes 4bit
################################################################################
use_4bit = True  # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
use_nested_quant = False  # Activate nested quantization for 4-bit base models (double quantization)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

print("************************* Load Model *************************")

################################################################################
# merge Model
################################################################################

config = PeftConfig.from_pretrained(
    new_peft_model
)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map=device_map,
    output_hidden_states=True
)
model = PeftModel.from_pretrained(model, new_peft_model)
print("************************* Load Model Done*************************")

with open('dataset/build_data/dataset/contrastive/test.txt',
          'r') as file:
    datasets = [next(file) for _ in range(512)]

################################################################################
# Generate Data
################################################################################
print("************************* Generate Data *************************")
new_result = []

for encoded_inputs in datasets:
    batch_input = tokenizer.encode_plus(
        encoded_inputs,
        return_attention_mask=True,
        padding='max_length',
        max_length=32,
        return_tensors='pt'
    ).to(device)
    input_ids = batch_input["input_ids"]
    attention_mask = batch_input["attention_mask"]
    model.eval()


    with torch.no_grad():

        outputs = model(
            **batch_input,
            output_hidden_states=True
        )


        hidden_states = outputs.hidden_states

    last_layer_hidden_states = hidden_states[-1]

    sequence_output = (outputs.hidden_states[-1] + outputs.hidden_states[0]) / 2

    batch_emb = sequence_output.mean(dim=1)

    tensor_np = batch_emb.detach().cpu().numpy()
    new_result.append(np.squeeze(tensor_np))


np.save('mistral_contrastive_512x4096_.npy', new_result)
sys.exit(0)

print("************************* Generate Data Done *************************")
