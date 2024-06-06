import datetime

import torch
from transformers import BertTokenizer
from filter_model import Model
from data_loader import TextDataset, collate_fn

model_path = "bert-base-uncased /"
train_dataset_path = "classification/train_10000.csv"
test_dataset_path = "classification/test_2000.csv"
token = BertTokenizer.from_pretrained(model_path)


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(info + '...\n\n')


def test(threshold=0.5):
    correct = 0
    total = 0

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        correct += (((out >= threshold).float() * 1).flatten() == labels).sum().cpu().item()
        total += len(labels)
    accuracy = correct / total

    printlog(f"Eval: Threshold: {threshold}, acc: {accuracy:.10f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_dataset = TextDataset(train_dataset_path)
    test_dataset = TextDataset(test_dataset_path)

    batch_size = 1024

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True)

    criterion = torch.nn.BCELoss()
    model = Model()
    model.load_state_dict(
        torch.load('/media/data1/tfh/2023/09/fine_tuning/llama2_7B/post_process/bert_filter/4/model_best.pth')
    )
    model.to(device)
    model.eval()
    thresholds = [0.1, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7, 0.9, 0.99]
    for thre in thresholds:
        test(thre)
