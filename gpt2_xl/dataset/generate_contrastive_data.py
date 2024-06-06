import json
import random

COUNT = 100000
dataset_path = "../../dataset/conceptnet/rel_head2tails.json"
train_path = f"./contrastive/train.txt"
test_path = f"./contrastive/test_{COUNT * 0.001}.txt"

with open(dataset_path) as reader:
    rel_head2tails = json.load(reader)

relation_pairs = {
    "Antonym": "Antonym",
    "AtLocation": "LocatedAt",
    "CapableOf": "PerformedBy",
    "Causes": "CausedBy",
    "CausesDesire": "DesireCausedBy",
    "CreatedBy": "Creates",
    "DefinedAs": "Defines",
    "DerivedFrom": "DerivesInto",
    "Desires": "IsDesiredBy",
    "DistinctFrom": "DistinctFrom",
    "EtymologicallyDerivedFrom": "EtymologicallyGaveRiseTo",
    "EtymologicallyRelatedTo": "EtymologicallyRelatedTo",
    "FormOf": "IsFormOf",
    "HasA": "IsAOf",
    "HasContext": "IsContextOf",
    "HasFirstSubevent": "IsFirstSubeventOf",
    "HasLastSubevent": "IsLastSubeventOf",
    "HasPrerequisite": "IsPrerequisiteOf",
    "HasProperty": "IsPropertyOf",
    "HasSubevent": "IsSubeventOf",
    "IsA": "HasA",
    "LocatedNear": "LocatedNear",
    "MadeOf": "ComponentOf",
    "MannerOf": "ManifestationOf",
    "MotivatedByGoal": "EnablesGoal",
    "ObstructedBy": "Obstructs",
    "PartOf": "HasPart",
    "ReceivesAction": "PerformsAction",
    "RelatedTo": "RelatedTo",
    "SimilarTo": "SimilarTo",
    "SymbolOf": "SymbolizedBy",
    "Synonym": "Synonym",
    "UsedFor": "HasUsage"
}

relation_sentence = {
    "Antonym": "The antonym of [H] is [T].",
    "AtLocation": "[H] is typically located in [T].",
    "CapableOf": "[H] can perform [T].",
    "Causes": "It’s common for [H] to trigger [T].",
    "CausesDesire": "[H] tends to inspire a desire for [T].",
    "CreatedBy": "[H] was brought into existence by [T].",
    "DefinedAs": "[H] can be described as [T].",
    "DerivedFrom": "[H] is embedded in the meaning of the expression [T].",
    "Desires": "[H] has a desire for [T].",
    "DistinctFrom": "In the set, [H] is not the same as [T].",
    "EtymologicallyDerivedFrom": "The word [H] has its roots in the term [T].",
    "EtymologicallyRelatedTo": "There’s a common etymological origin for [H] and [T].",
    "FormOf": "[H] is a variant form of [T].",
    "HasA": "[H] is the owner of [T].",
    "HasContext": "The word [H] is recognized and utilized in the context of [T].",
    "HasFirstSubevent": "The journey of [H] kicks off with [T].",
    "HasLastSubevent": "The final score of [H] is always [T].",
    "HasPrerequisite": "[H] is dependent on [T].",
    "HasProperty": "[H] can be characterized by [T].",
    "HasSubevent": "During the course of [H], you might experience [T].",
    "IsA": "[H] is an example of [T].",
    "LocatedNear": "[H] is situated close to [T].",
    "MadeOf": "[H] consists of [T].",
    "MannerOf": "[H] represents a particular method of executing [T].",
    "MotivatedByGoal": "[H] is a means to the end of [T].",
    "ObstructedBy": "[H] is obstructed by [T].",
    "PartOf": "[H] belongs to [T].",
    "ReceivesAction": "[H] has the ability to undergo [T].",
    "RelatedTo": "[H] has a connection with [T].",
    "SimilarTo": "[H] bears resemblance to [T].",
    "SymbolOf": "[H] stands for [T].",
    "Synonym": "[H] is synonymous with [T].",
    "UsedFor": "The purpose of [H] is to achieve [T]."
}

reverse_realtion_sentence = {
    "Antonym": "The antonym of [H] is [T].",
    "LocatedAt": "The place known for the presence of [H] is [T].",
    "PerformedBy": "[H] is an activity that was completed by [T].",
    "CausedBy": "The reason behind [H] has been identified as [T].",
    "DesireCausedBy": "The reason behind [H] has been identified as [T].",
    "Creates": "[H] is the originator of [T].",
    "Defines": "[H] serves to clarify what is meant by [T].",
    "DerivesInto": "The concept of [H] evolves and takes on a new form as [T].",
    "IsDesiredBy": "[H] finds itself in the favorable position of being needed by [T].",
    "DistinctFrom": "In the set, [H] is not the same as [T].",
    "EtymologicallyGaveRiseTo": "The term [H] has historically evolved and notably given rise to the contemporary term [T].",
    "EtymologicallyRelatedTo": "There’s a common etymological origin for [H] and [T].",
    "IsFormOf": "The word [H] stems from its basic form [T].",
    "IsAOf": "[H] falls under the category of [T].",
    "IsContextOf": "The subject of [H] is inherently linked to the idea of [T]. ",
    "IsFirstSubeventOf": "Serving as the precursor, [H] paves the way for the unfolding of [T].",
    "IsLastSubeventOf": "[H] serves as the ultimate phase in the progression of [T].",
    "IsPrerequisiteOf": "Mastery of [H] is a fundamental requirement for anyone aiming to reach [T].",
    "IsPropertyOf": "The attribute [H] can be identified as belonging uniquely to [T].",
    "IsSubeventOf": "[H] unfolds as an integral part of the grander event known as [T].",
    "HasA": "Among its various properties, [H] includes a [T] in its composition.",
    "LocatedNear": "[H] is situated close to [T].",
    "ComponentOf": "[H] is an integral part of the larger structure that is [T].",
    "ManifestationOf": "The presence of [H] is indicative of the underlying concept of [T].",
    "EnablesGoal": "The action of [H] is instrumental in bringing about the ultimate goal of [T].",
    "Obstructs": "[H] imposes significant barriers that restrict the development of [T].",
    "HasPart": "[H] is composed of several components, including [T].",
    "PerformsAction": "[H] actively engages in the process of [T].",
    "RelatedTo": "[H] has a connection with [T].",
    "SimilarTo": "[H] bears resemblance to [T].",
    "SymbolizedBy": "[H] is commonly used to represent [T].",
    "Synonym": "[H] is synonymous with [T].",
    "HasUsage": "[H] is designed with the capability to serve as [T]."
}


def random_select(dictionary, num_elements):
    """
    :param dictionary:
    :param num_elements:
    :return:
    """
    keys = list(dictionary.keys())
    selected_keys = random.sample(keys, num_elements)
    selected_elements = {key: dictionary[key] for key in selected_keys}
    return selected_elements


def get_relation_sentence(rel, head, tail, reverse=False):
    """
    :param rel:
    :param head:
    :param tail:
    :param reverse:
    :return:
    """
    if reverse:
        return reverse_realtion_sentence[rel].replace("[H]", head).replace("[T]", tail).replace("_", " ").strip()
    else:
        return relation_sentence[rel].replace("[H]", head).replace("[T]", tail).replace("_", " ").strip()


def generate_data(data=None):
    all_data = []
    train_data = []
    test_data = []


    for rel_head, tails in data.items():
        rel, head = rel_head.split("-")

        for tail in tails:
            all_data.append({
                "rel": rel.strip(),
                "head": head.strip(),
                "tail": tail.strip(),
                "rev_rel": relation_pairs[rel].strip()
            })

    random.shuffle(all_data)
    random_train = all_data[:int(len(all_data) * 0.999)]
    random_test = all_data[int(len(all_data) * 0.999):]

    for item in random_train:
        rel, head, tail, rev_rel = item.items()
        train_data.append(get_relation_sentence(rel[1], head[1], tail[1]))
        train_data.append(get_relation_sentence(rev_rel[1], tail[1], head[1], True))
    for item in random_test:
        rel, head, tail, rev_rel = item.items()
        test_data.append(get_relation_sentence(rel[1], head[1], tail[1]))
        test_data.append(get_relation_sentence(rev_rel[1], tail[1], head[1], True))

    with open(train_path, 'w') as file:
        file.write('\n'.join(train_data))
    with open(test_path, 'w') as file:
        file.write('\n'.join(test_data))

if __name__ == "__main__":
    generate_data(rel_head2tails)
