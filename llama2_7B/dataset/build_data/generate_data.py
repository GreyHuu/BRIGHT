import json
import random
from get_instruction import get_instruction

COUNT = 2000
TEST_COUNT = int(COUNT * 0.2)
dataset_path = "dataset/rel_head2tails.json"
sim_dataset_path = "dataset/rel_head2sims.json"
train_path = f"./dataset/relation/train_{COUNT}.txt"
test_path = f"./dataset/relation/test_{TEST_COUNT}.txt"

with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)

with open(sim_dataset_path) as reader:
    rel_head2sims = json.load(reader)


def get_output_by_input(relation, head):


    relation_head = relation + "-" + head
    try:
        result = rel_head2tails[relation_head]
    except KeyError:
        return None
    return result[:5]


def get_sim_input_output(relation, head):

    relation_head = relation + "-" + head

    try:
        sim_heads = rel_head2sims[relation_head]
    except KeyError:
        return None

    sim_input_output = []


    # for sim_head in sim_heads:
    #     sim_input_output.append({
    #         "input": sim_head,
    #         "output": get_output_by_input(relation, sim_head)
    #     })

    return sim_input_output[:5]


def random_select(dictionary, num_elements):

    keys = list(dictionary.keys())
    selected_keys = random.sample(keys, num_elements)
    selected_elements = {key: dictionary[key] for key in selected_keys}
    return selected_elements


def generate_data(rel_head2tails=None):
    all_data = []
    train_data = []
    test_data = []

    rel_head2tails = {key: rel_head2tails[key] for key, item in rel_head2tails.items() if len(item) > 2}

    rel_head2tails = random_select(rel_head2tails, COUNT + TEST_COUNT)

    for rel_head, tails in rel_head2tails.items():
        rel, head = rel_head.split("-")
        sim_head_tails = get_sim_input_output(rel, head)
        tails = tails[:5]
        if rel is not None and head is not None and tails is not None and sim_head_tails is not None:
            all_data.append({
                "rel": rel,
                "head": head,
                "tails": tails,
                "sim_head_tails": sim_head_tails
            })
        else:
            print("none")
    #
    random_train_1000 = random.sample(all_data, COUNT)
    #
    random_test_200 = [x for x in all_data if x not in random_train_1000]



    for item in random_train_1000:
        rel, head, tails, sim_head_tails = item.items()
        train_data.append(get_instruction(rel[1], head[1], tails[1], sim_head_tails[1]))

    for item in random_test_200:
        rel, head, tails, sim_head_tails = item.items()
        test_data.append(get_instruction(rel[1], head[1], tails[1], sim_head_tails[1]))

    with open(train_path, 'w') as file:
        for item in train_data:
            file.write(item + '\n')
    with open(test_path, 'w') as file:
        for item in test_data:
            file.write(item + '\n')


if __name__ == "__main__":
    generate_data(rel_head2tails)
