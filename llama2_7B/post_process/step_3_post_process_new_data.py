import copy
import json
import os
import threading
from tqdm import tqdm
from gpt_score import get_score_by_gpt
from collections import Counter
from novelty_sim import get_sim_by_embedding, get_embedding_by_gpt
from joblib import Parallel, delayed


file_lock = threading.Lock()
dataset_path = "dataset/rel_head2tails.json"
with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)

relation_sentence = {
    "Antonym": "The antonym of [H] is [T]",
    "AtLocation": "[H] is typically located in [T]",
    "CapableOf": "[H] can perform [T]",
    "Causes": "It’s common for [H] to trigger [T]",
    "CausesDesire": "[H] tends to inspire a desire for [T]",
    "CreatedBy": "[H] was brought into existence by [T]",
    "DefinedAs": "[H] can be described as [T]",
    "DerivedFrom": "[H] is embedded in the meaning of the expression [T]",
    "Desires": "[H] has a desire for [T]",
    "DistinctFrom": "In the set, [H] is not the same as [T]",
    "EtymologicallyDerivedFrom": "The word [H] has its roots in the term [T]",
    "EtymologicallyRelatedTo": "There’s a common etymological origin for [H] and [T]",
    "FormOf": "[H] is a variant form of [T]",
    "HasA": "[H] is the owner of [T]",
    "HasContext": "The word [H] is recognized and utilized in the context of [T]",
    "HasFirstSubevent": "The journey of [H] kicks off with [T]",
    "HasLastSubevent": "The final score of [H] is always [T]",
    "HasPrerequisite": "[H] is dependent on [T]",
    "HasProperty": "[H] can be characterized by [T]",
    "HasSubevent": "During the course of [H], you might experience [T]",
    "IsA": "[H] is an example of [T]",
    "LocatedNear": "[H] is situated close to [T]",
    "MadeOf": "[H] consists of [T]",
    "MannerOf": "[H] represents a particular method of executing [T]",
    "MotivatedByGoal": "[H] is a means to the end of [T]",
    "ObstructedBy": "[H] is obstructed by [T]",
    "PartOf": "[H] belongs to [T]",
    "ReceivesAction": "[H] has the ability to undergo [T]",
    "RelatedTo": "[H] has a connection with [T]",
    "SimilarTo": "[H] bears resemblance to [T]",
    "SymbolOf": "[H] stands for [T]",
    "Synonym": "[H] is synonymous with [T]",
    "UsedFor": "The purpose of [H] is to achieve [T]"
}


def save_json_file(json_data, path):
    with file_lock:
        with open(path, "w") as writer:
            writer.write(json.dumps(json_data))


def get_tails_by_rel_head(relation, head):
    relation_head = relation + "-" + head
    try:
        result = rel_head2tails[relation_head]
    except KeyError:
        return []
    return result


def get_cos_sim(answer, tails, similar_threshold=.95):
    return get_sim_by_embedding(answer.replace("_", " "), tails, threshold=similar_threshold)


def get_novelty(data, data_name):
    total_len = 0
    novelty_len = 0
    data = copy.deepcopy(data)
    for i, _ in tqdm(enumerate(data), desc="Novelty Calculation", total=len(data)):
        item = data[i]
        rel = item["rel"]
        head = item["head"]
        answers = item["filtered_answers"]
        tails = get_tails_by_rel_head(rel, head)
        str_tails = [item.replace("_", " ") for item in tails]
        embedding_tails = get_embedding_by_gpt(str_tails)
        current_len = len(answers)

        try:
            if isinstance(answers[0], dict):
                similar_indices = Parallel(n_jobs=-1)(
                    delayed(get_cos_sim)(element["answer"], embedding_tails) for element in answers
                )
                novelty_tails = [answers[i] for i, similar in enumerate(similar_indices) if not similar]
            else:

                similar_indices = Parallel(n_jobs=-1)(
                    delayed(get_cos_sim)(element, embedding_tails) for element in answers
                )
                novelty_tails = [answers[i] for i, similar in enumerate(similar_indices) if not similar]

            item.update({
                "answers": novelty_tails
            })
            data[i] = item
            novelty_len += len(novelty_tails)
            total_len += current_len
            save_json_file(data, f"./score/score_novelty_{data_name}")
        except Exception as e:
            print(e)
    return data, novelty_len / total_len * 100.0


def find_dominant_element(lst):
    counts = Counter(lst)
    for element, count in counts.items():
        if count >= 2:
            return element
    return "3"


def thread_get_score(thread_index, score_index, all_data, start_index, end_index, data_name):
    print(f"""
             Thread {thread_index}  Parameters：
                    "score_index": {score_index},
                    "start_index": {start_index},
                    "end_index": {end_index}
    """)
    for i in range(start_index, end_index):
        item = all_data[i]
        rel = item["rel"]
        head = item["head"]

        answers = item["filtered_answers"]
        all_scores = []
        for j in range(0, len(answers), 5):
            tails = answers[j:j + 5]
            rels = [rel] * len(tails)
            heads = [head] * len(tails)
            if isinstance(tails[0], str):
                sentences = [relation_sentence[rel].replace("[H]", head).replace("[T]", item) + ". " for item in tails]
            else:
                sentences = [relation_sentence[rel].replace("[H]", head).replace("[T]", item["answer"]) + ". " for item
                             in tails]
            scores = get_score_by_gpt(rels, heads, tails, sentences)
            print(scores)
            for item in scores:
                all_scores.append(item)
        if len(answers) != len(all_scores):
            return [], 0
        else:
            new_answers = []
            for temp in range(len(answers)):
                if isinstance(answers[0], dict):
                    new_answers.append({
                        **answers[temp],
                        f"score_{score_index}": all_scores[temp]
                    })
                else:
                    new_answers.append({
                        "answer": answers[temp],
                        f"score_{score_index}": all_scores[temp]
                    })
            all_data[i]["filtered_answers"] = new_answers
            save_temp_data = all_data[start_index:end_index]
            save_json_file(save_temp_data,
                           f"./score/temp/{thread_index}_score_data_{data_name}")


def merge_data(data_list):
    m_data = []
    for item in data_list:
        with open(item, "r") as file:
            m_data.append(json.load(file))
        os.remove(item)
    m_data = [item for sublist in m_data for item in sublist]
    return m_data


def divide_list_into_parts(lst, n=5):
    part_size = len(lst) // n
    indexes = [(i * part_size, (i + 1) * part_size) for i in range(n)]
    indexes[-1] = (indexes[-1][0], len(lst))
    return indexes


def gpt_4_get_un_score(temp_gpt4_list):
    rels = []
    heads = []
    tails = []
    sentences = []
    for temp_item in temp_gpt4_list:
        item = temp_item["item"]
        answer = temp_item["answer"]
        rel = item["rel"]
        head = item["head"]

        rels.append(rel)
        heads.append(head)
        tails.append(answer)
        sentences.append(relation_sentence[rel].replace("[H]", head).replace("[T]", answer) + ". ")
    score = get_score_by_gpt(rels, heads, tails, sentences, flag=False,
                             model_name="gpt-4-0125-preview")
    print("gpt-4: ", score)
    return score


def get_user_input(rel, head, tail):
    while True:
        user_input = input(f"Head: {head}, Relation: {rel}, Answer: {tail},Please decide and then enter Y(y) or N(n).: ").strip().upper()
        if user_input == 'Y' or user_input == 'N':
            return user_input
        else:
            print("Invalid input. Please enter Y or N again.")


def get_acc(data, temp_data_name):
    data = copy.deepcopy(data)
    data_name = temp_data_name

    gpt_times = 3
    thread_shard_data_count = 5
    thread_shard_data_index = divide_list_into_parts(data, thread_shard_data_count)

    for index in range(gpt_times):
        thread_list = []
        for thread_index in range(thread_shard_data_count):
            thread = threading.Thread(
                target=thread_get_score,
                kwargs={
                    "thread_index": thread_index,
                    "score_index": index,
                    "all_data": data,
                    "start_index": thread_shard_data_index[thread_index][0],
                    "end_index": thread_shard_data_index[thread_index][1],
                    "data_name": temp_data_name
                })

            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
        shard_data_path_list = [
            f"./score/temp/{thread_index}_score_data_{data_name}"
            for thread_index in range(thread_shard_data_count)]
        data = merge_data(shard_data_path_list)

    total_count = 0
    yes_count = 0
    no_count = 0
    un_count = 0

    for index in range(len(data)):
        item = data[index]
        answers = item["filtered_answers"]
        total_count += len(answers)
        temp_gpt4_list = []
        for j in range(len(answers)):
            i = answers[j]
            score_list = [i[f"score_{index}"] for index in range(gpt_times)]
            result_score = find_dominant_element(score_list)

            if result_score == "Unidentifiable":
                temp_gpt4_list.append({
                    "index": j,
                    "item": item,
                    "answer": i["answer"]
                })

            if result_score == "Yes":
                yes_count += 1
            elif result_score == "No":
                no_count += 1
            max_length = 5
            if (len(temp_gpt4_list) >= max_length) or (len(temp_gpt4_list) > 0 and j == len(answers) - 1):
                scores = gpt_4_get_un_score(temp_gpt4_list)
                for t_item, t_score in zip(temp_gpt4_list, scores):
                    data[index]["filtered_answers"][t_item["index"]].update({
                        "result_score": t_score
                    })
                temp_gpt4_list = []
            else:
                data[index]["filtered_answers"][j].update({
                    "result_score": result_score
                })
            save_json_file(data, f"./score/score_novelty_{data_name}")

    print(f"""
        total_count :{total_count}
        yes_count :{yes_count}
        no_count :{no_count}
        un_count :{un_count}
    """)
    return data, yes_count / (yes_count + no_count) * 100.0


def process_acc_nov(data, temp_data_name):
    new_data = data
    print("## Starting novelty calculation:")

    novelty_data, novelty = get_novelty(new_data, temp_data_name)

    print("Novelty calculation completed ## ")
    print("## Starting accuracy calculation: ")

    acc_data, acc = get_acc(new_data, temp_data_name)

    print("Accuracy calculation is complete. ## ")
    print("## Begin calculating the novelty in the correct dataset: ")
    _, acc_novelty = get_novelty(acc_data,temp_data_name)
    print("The calculation of novelty in the correct dataset has been completed. ## ")
    return f"""
        Data (Data Name):{temp_data_name}
        Novelty (All Data): {novelty:.2f}%
        Accuracy (All Data): {acc:.2f}%
        Novelty (accurate data): {acc_novelty:.2f}%
    """
