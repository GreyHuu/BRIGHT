import json

data_path = "filtered_llama-2-7b-llama_without_contrast_10k_8_1_1_new_data_200.json"

with open(data_path, "r") as file:
    dataset = json.load(file)

answers_count = 0
filtered_answer_count = 0

for temp in dataset:
    answers = temp["answers"]
    filtered_answer = temp["filtered_answers"]
    answers_count += len(answers)
    filtered_answer_count += len(filtered_answer)

print("filtered: ", filtered_answer_count, "; generate: ", answers_count, "; acc: ",
      filtered_answer_count / answers_count)
