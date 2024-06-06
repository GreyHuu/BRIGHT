import json

from bert_filter.filter_func import handle_filter_answer

def post_process(data):
    for item in data:
        answer = item["answers"]
        n_answer = []
        for j in answer:
            for i in j:
                if len(i.split("_")) <= 5:
                    n_answer.append(i)
        item.update({
            "answers": list(set(n_answer))
        })
    return data


def filter_new_data(data=None, filtered_data_path=None):
    data = post_process(data)
    for i in range(len(data)):
        item = data[i]
        rel = item["rel"]
        head = item["head"]
        answers = item["answers"]
        filtered_new_answer = []
        for tail in answers:
            if handle_filter_answer(rel, head, tail):
                filtered_new_answer.append(tail)
        data[i].update({
            "filtered_answers": filtered_new_answer
        })
    with open(filtered_data_path, "w") as file:
        file.write(json.dumps(data))


def count_filtered(dataset):
    answers_count = 0
    filtered_answer_count = 0

    for temp in dataset:
        answers = temp["answers"]
        filtered_answer = temp["filtered_answers"]
        answers_count += len(answers)
        filtered_answer_count += len(filtered_answer)

    return f"filtered number: {filtered_answer_count} ; generate number: {answers_count} ; bert_filtered_acc: {filtered_answer_count / answers_count}"
