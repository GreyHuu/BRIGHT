import json

COUNT = 1000
TEST_COUNT = int(COUNT * 0.2)
dataset_path = "dataset/rel_head2tails.json"
reverse_dataset_path = "dataset/reverse_rel_head2tails.json"
train_path = f"./reverse_train_{COUNT}.txt"
test_path = f"./reverse_test_{TEST_COUNT}.txt"

with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)

reverse_relation_pairs = {
    "AtLocation": "LocatedAt",
    "CapableOf": "PerformedBy",
    "Causes": "PerformedBy",
    "CausesDesire": "DesireCausedBy",
    "CreatedBy": "Creates",
    "DefinedAs": "Defines",
    "DerivedFrom": "DerivesInto",
    "Desires": "IsDesiredBy",
    "EtymologicallyDerivedFrom": "EtymologicallyGaveRiseTo",
    "FormOf": "IsFormOf",
    "HasA": "IsAOf",
    "HasContext": "IsContextOf ",
    "HasFirstSubevent": "IsFirstSubeventOf",
    "HasLastSubevent": "IsLastSubeventOf",
    "HasPrerequisite": "IsPrerequisiteOf",
    "HasProperty": "IsPropertyOf",
    "HasSubevent": "IsSubeventOf",
    "IsA": "ParentOf",
    "MadeOf": "ComponentOf",
    "MannerOf": "ManifestationOf",
    "MotivatedByGoal": "EnablesGoal",
    "ObstructedBy": "Obstructs",
    "PartOf": "HasPart",
    "ReceivesAction": "PerformsAction",
    "SymbolOf": "SymbolizedBy",
    "UsedFor": "HasUsage"
}


def find_element_index(element, lst):
    try:
        index = lst.index(element)
        return index
    except ValueError:
        return None


def generate_reverse_relation():
    relations = list(reverse_relation_pairs.keys())
    reverse_relations = list(reverse_relation_pairs.values())

    reverse_data = {}
    count = 0
    for rel_head, tails in rel_head2tails.items():
        rel, head = rel_head.split("-")
        index = find_element_index(rel, relations)
        if index:
            reverse_rel = reverse_relations[index]
        else:
            continue
        for tail in tails:
            rev_rel_head = reverse_rel + "-" + tail

            rev_tails = reverse_data.get(rev_rel_head, [])
            rev_tails.append(head)
            reverse_data[rev_rel_head] = rev_tails

    with open(reverse_dataset_path, 'w') as file:
        file.write(json.dumps(reverse_data))


if __name__ == "__main__":
    generate_reverse_relation()
