import json
import os
import re

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
from torch.utils.data import DataLoader
from trl import SFTTrainer

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
COUNT = 2000
dataset_path = f"dataset/for_new_datas/generate_{COUNT}.json"
new_model_base_path = "mistral/new_model/"
peft_model_id = "mistral-7b_chat_contrast_10k_8_1_1data_3epoch_ATOMIC"
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
# bitsandbytes
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

################################################################################
################################################################################
print("************************* Load Data *************************")

datasets = load_dataset("json", data_files=dataset_path)["train"]
# with open(dataset_path, "r") as reader:
#     datasets = json.load(reader)

print("************************* Load Data Done *************************")


################################################################################
#
################################################################################
#
# force_words = ["Output entities:"]
# force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

def post_process(answer):
    """

    :param answer:
    :return:
    """
    answer = answer.strip()
    key_word = "Output nodes:"
    if answer.startswith(key_word):

        answer = answer.replace(key_word, "").strip()
        split_answer = re.split(r' +', answer)
        #
        answer = split_answer[0].split(",")
        return answer
    else:
        return []


################################################################################
# Generate Data
################################################################################
print("************************* Generate Data *************************")
new_result = []
for step in range(0, len(datasets), batch_size):
    encoded_inputs = datasets[step:step + batch_size]
    batch_input = tokenizer.batch_encode_plus(
        encoded_inputs["prompt"],
        padding="longest",
        return_tensors='pt'
    ).to(device)
    input_ids = batch_input["input_ids"]
    attention_mask = batch_input["attention_mask"]
    outputs = model.generate(
        **batch_input,
        do_sample=False,
        num_beams=num_sentence,
        num_beam_groups=num_sentence,
        diversity_penalty=diversity_penalty,
        max_new_tokens=max_new_token,
        num_return_sequences=num_sentence,
        # force_words_ids=force_words_ids,
        early_stopping=True,
        temperature=1,
        top_p=1
    )
    batch_size, seq_len = input_ids.size()

    for j in range(batch_size):
        current_data = datasets[step + j]
        rel = current_data["rel"]
        head = current_data["head"]
        index = current_data["index"]
        print("ITEM PROCESS: " + rel + "-" + head)
        nums_answers = []
        # for t in outputs:
        #     te_item = tokenizer.decode(t)
        for i in range(num_sentence):

            item = outputs[j * batch_size + i].cpu()

            ids = item[seq_len:].numpy().tolist()
            #
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            #
            answer = post_process(answer)
            print(answer)
            nums_answers.append(answer)
        new_result.append({
            "index": index,
            "rel": rel,
            "head": head,
            "answers": nums_answers
        })

with open(save_data_path, "w") as file:
    file.write(json.dumps(new_result))
print("************************* Generate Data Done *************************")
