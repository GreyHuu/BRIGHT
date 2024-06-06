import json
from step_1_filter_answer import filter_new_data, count_filtered
from step_3_post_process_new_data import process_acc_nov

new_data_base = "gpt_2XL/new_datas/"

data_name = "gpt2-xl_10k_8_1_1data_3epoch_new_data_20_sentence_10_penalty_1.0.json"

dataset_path = new_data_base + data_name


filtered_data_path = f"gpt_2XL/new_datas/filtered/filtered_{data_name}"


def step_1_filter_answer(data, save_path):
    filter_new_data(data, save_path)


def step_2_filter_result(data):
    return count_filtered(data)


def step_3_acc_nov(data, name):
    return process_acc_nov(data, name)


if __name__ == "__main__":
    print("************************* Start filtering *************************")
    with open(dataset_path, "r") as file:
        filter_datas = json.load(file)

    step_1_filter_answer(filter_datas, filtered_data_path)

    with open(filtered_data_path, "r") as file:
        filtered_data = json.load(file)

    bert_filtered_result = step_2_filter_result(filtered_data)
    print("************************* Filtration complete *************************")
    print("************************* Begin accuracy and novelty evaluation *************************")
    result = step_3_acc_nov(filtered_data, data_name)
    print("************************* End of evaluation *************************")
    print("************************* The results are as follows:_ *************************")
    print("Filter Model:\n", bert_filtered_result + "\n")
    print("Eval Model:\n", result + "\n")
