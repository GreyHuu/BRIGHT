import json
import random
from get_cls_instruction import get_instruction

COUNT = 10000
TEST_COUNT = int(COUNT * 0.2)
dataset_path = "dataset/classification_rel_head2tails.json"
train_path = f"./dataset/classification/train_{COUNT}.txt"
test_path = f"./dataset/classification/test_{TEST_COUNT}.txt"

with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)

with open(dataset_path) as reader:
    rel_head2sims = json.load(reader)


def get_output_by_input(relation, head):
    relation_head = relation + "-" + head
    try:
        result = rel_head2tails[relation_head]
    except KeyError:
        return None
    return result[:5]


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


def shuffle_model_answer(tails, fake_tails):
    output = tails + fake_tails


    random.shuffle(output)

    reasonable = [i for i in output if i in tails]
    unreasonable = [i for i in output if i in fake_tails]

    return output, reasonable, unreasonable


def generate_data(rel_head2tails=None):
    all_data = []
    train_data = []
    test_data = []

    rel_head2tails = [item for item in rel_head2tails if len(item["tails"]) > 2]
    rel_head2tails = random_select(rel_head2tails, COUNT + TEST_COUNT)

    """
            "rel": rel,
            "head": head,
            "tails": tails,
            "fake_tails": fake_tails
    """
    for current_item in rel_head2tails:
        rel, head, tails, fake_tails = return_sub_obj(current_item)

        if rel is not None and head is not None and tails is not None and fake_tails is not None:
            all_data.append({
                "rel": rel,
                "head": head,
                "tails": tails,
                "fake_tails": fake_tails
            })
        else:
            print("none")

    # train
    random_train_1000 = random.sample(all_data, COUNT)
    # test
    random_test_200 = [x for x in all_data if x not in random_train_1000]

    for item in random_train_1000:
        rel, head, tails, fake_tails = return_sub_obj(item)
        output, reasonable, unreasonable = shuffle_model_answer(tails, fake_tails)
        train_data.append(get_instruction(rel, head, output, reasonable, unreasonable))

    for item in random_test_200:
        rel, head, tails, fake_tails = return_sub_obj(item)
        output, reasonable, unreasonable = shuffle_model_answer(tails, fake_tails)
        test_data.append(get_instruction(rel, head, output, reasonable, unreasonable))

    with open(train_path, 'w') as file:
        for item in train_data:
            file.write(item + '\n')
    with open(test_path, 'w') as file:
        for item in test_data:
            file.write(item + '\n')


if __name__ == "__main__":
    generate_data(rel_head2tails)
