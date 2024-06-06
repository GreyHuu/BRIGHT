import json
import random

COUNT = 1000
TEST_COUNT = int(COUNT * 0.2)
dataset_path = "dataset/rel_head2tails.json"
cls_dataset_path = "dataset/classification_rel_head2tails.json"

with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)


def get_random_key(dictionary):
    print("key")
    keys = list(dictionary.keys())
    random_key = random.choice(keys)
    return random_key


def generate_classification_relation():
    cls_rel_head2tails = []
    keys = list(rel_head2tails.keys())
    # count = 0
    for rel_head, tails in rel_head2tails.items():
        print(rel_head)
        rel, head = rel_head.split("-")
        tails = tails[:5]
        # random_element = get_random_key(rel_head2tails)
        random_key = random.choice(keys)
        while random_key == rel_head:
            print("while")
            random_key = random.choice(keys)
        fake_tails = rel_head2tails[random_key][:5]
        cls_rel_head2tails.append({
            "rel": rel,
            "head": head,
            "tails": tails,
            "fake_tails": fake_tails
        })
        # if count > 1000:
        #     break
        # count += 1
    with open(cls_dataset_path, 'w') as file:
        file.write(json.dumps(cls_rel_head2tails))


if __name__ == "__main__":
    generate_classification_relation()
