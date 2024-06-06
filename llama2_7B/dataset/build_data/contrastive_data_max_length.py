import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

model_name = "/llama2_7B_hf/"
data_path = "dataset/build_data/dataset/contrastive/train.txt"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

with open(data_path, "r", encoding='utf-8') as file:
    lines = file.readlines()

tok_length_list = []
for line in lines:
    line = line.strip()
    tok_length_list.append(len(tokenizer.encode(line)))



p80 = np.percentile(tok_length_list, 80)
p90 = np.percentile(tok_length_list, 90)
p99 = np.percentile(tok_length_list, 99)

mean_value = np.mean(tok_length_list)


print(f"80: {p80}")
print(f"90: {p90}")
print(f"99: {p99}")
print(f"mean: {mean_value}")