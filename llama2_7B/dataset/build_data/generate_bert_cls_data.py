import json
import random
from joblib import Parallel, delayed

COUNT = 1600000
TEST_COUNT = int(COUNT * 0.05)
dataset_path = "dataset/classification_rel_head2tails.json"
train_path = f"./dataset/classification/train_{COUNT}.csv"
test_path = f"./dataset/classification/test_{TEST_COUNT}.csv"

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


def random_select(dictionary, num_elements):
    random_elements = random.sample(dictionary, num_elements)
    return random_elements


def return_sub_obj(current_item):
    rel = current_item["rel"]
    head = current_item["head"]
    tails = current_item["tails"]
    fake_tails = current_item["fake_tails"]

    tails = tails[:5]
    fake_tails = fake_tails[:5]

    return rel, head, tails, fake_tails


def return_bert_cls_csv(rel, head, tail, is_right):
    if is_right:
        is_right = 1
    else:
        is_right = 0
    return f'{is_right},Triple:Head entity: "{head.replace(",", " ")}" Relationship: "{rel}" Tail entity: "{tail.replace(",", " ")}". Sentence: {relation_sentence[rel].replace("[H]", head).replace("[T]", tail).replace(",", " ")}.'


def process_item(item):
    data = []
    rel = item["rel"]
    head = item["head"]
    tails = item["tails"]
    fake_tails = item["fake_tails"]
    # rel, head, tails, fake_tails = return_sub_obj(item)
    for temp in tails:
        data.append(return_bert_cls_csv(rel, head, temp.replace(",", " "), True))
    for temp in fake_tails:
        data.append(return_bert_cls_csv(rel, head, temp.replace(",", " "), False))

    return data


def generate_data_parallel(rel_head2tails=None):
    train_data = ["label,text"]
    test_data = ["label,text"]
    # random_train = random.sample(rel_head2tails, COUNT)
    # random_test = [x for x in rel_head2tails if x not in random_train]
    random_train = rel_head2tails[:int(len(rel_head2tails) * 0.95)]
    random_test = rel_head2tails[int(len(rel_head2tails) * 0.95):]
    train_results = Parallel(n_jobs=-1)(
        delayed(process_item)(temp) for temp in random_train
    )

    test_results = Parallel(n_jobs=-1)(
        delayed(process_item)(temp) for temp in random_test
    )

    for result in train_results:
        train_data.extend(result)

    for result in test_results:
        test_data.extend(result)

    with open(train_path, 'w') as file:
        for item in train_data:
            file.write(item + '\n')
    with open(test_path, 'w') as file:
        for item in test_data:
            file.write(item + '\n')


if __name__ == "__main__":
    generate_data_parallel(rel_head2tails)
