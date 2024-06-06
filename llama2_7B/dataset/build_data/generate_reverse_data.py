import json
import random
from get_instruction import get_instruction

COUNT = 4000
TEST_COUNT = int(COUNT * 0.2)
dataset_path = "dataset/reverse_rel_head2tails.json"
train_path = f"./dataset/reverse/reverse_train_{COUNT}.txt"
test_path = f"./dataset/reverse/reverse_test_{TEST_COUNT}.txt"

with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)


def get_output_by_input(relation, head):
    relation_head = relation + "-" + head
    try:
        result = rel_head2tails[relation_head]
    except KeyError:
        return None
    return result


def random_select(dictionary, num_elements):
    keys = list(dictionary.keys())
    selected_keys = random.sample(keys, num_elements)
    selected_elements = {key: dictionary[key] for key in selected_keys}
    return selected_elements


def generate_data(reverse_rel_head2tails=None):
    all_data = []
    train_data = []
    test_data = []

    reverse_rel_head2tails = random_select(reverse_rel_head2tails, COUNT + TEST_COUNT)

    for rel_head, tails in reverse_rel_head2tails.items():
        rel, head = rel_head.split("-")
        tails = tails[:5]

        all_data.append({
            "rel": rel,
            "head": head,
            "tails": tails
        })

    #
    random_train_1000 = random.sample(all_data, COUNT)
    #
    random_test_200 = [x for x in all_data if x not in random_train_1000]

    for item in random_train_1000:
        rel, head, tails = item.items()
        train_data.append(get_instruction(rel[1], head[1], tails[1], []))

    for item in random_test_200:
        rel, head, tails = item.items()
        test_data.append(get_instruction(rel[1], head[1], tails[1], []))

    with open(train_path, 'w') as file:
        for item in train_data:
            file.write(item + '\n')
    with open(test_path, 'w') as file:
        for item in test_data:
            file.write(item + '\n')


if __name__ == "__main__":
    generate_data(rel_head2tails)
