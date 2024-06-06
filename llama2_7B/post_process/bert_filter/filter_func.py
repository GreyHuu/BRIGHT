import torch
from transformers import BertTokenizer
from .filter_model import Model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Load Bert Filter Model")
model = Model()
model.load_state_dict(
    torch.load('model_best.pth'))
model.to(device)
model.eval()
print("Load Bert Filter Model Done")

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


def handle_filter_answer(rel, head, tail, threshold=0.482):
    text = f'Triple:Head entity: "{head.replace(",", " ")}" Relationship: "{rel}" Tail entity: "{tail.replace(",", " ")}". Sentence: {relation_sentence[rel].replace("[H]", head).replace("[T]", tail).replace(",", " ")}.'
    token = tokenizer(
        text,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt')
    input_ids = token["input_ids"].to(device)
    attention_mask = token["attention_mask"].to(device)
    token_type_ids = token["token_type_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

    print(out, "  ", out >= threshold)
    return out >= threshold
