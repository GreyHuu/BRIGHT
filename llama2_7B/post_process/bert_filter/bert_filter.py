import datetime
import wandb
import torch
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from filter_model import Model
from data_loader import TextDataset, collate_fn

model_path = "bert-base-uncased"
train_dataset_path = "classification/train_1600000.csv"
test_dataset_path = "classification/test_80000.csv"
token = BertTokenizer.from_pretrained(model_path)

best_accuracy = 0


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(info + '...\n\n')


def test(step, epoch):
    """
    Eval
    :param step:
    :param epoch:
    :return:
    """
    global best_accuracy
    model.eval()
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

        correct += (((out >= 0.5).float() * 1).flatten() == labels).sum().cpu().item()
        total += len(labels)

    accuracy = correct / total

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), f"model_best.pth")
        best_accuracy = accuracy

    wandb.log({
        "epoch": epoch,
        "score": step,
        "eval_accuracy": accuracy
    })


def train():
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    optimizer = AdamW(model.parameters(), lr=3e-5)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * epochs * 0.03,
        num_training_steps=len(train_loader) * epochs)

    for epoch in range(epochs):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

            loss = criterion(out.view(-1), labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            accuracy = (((out >= 0.5).float() * 1).flatten() == labels).sum().cpu().item() / len(labels)

            wandb.log({
                "epoch": epoch,
                "score": i,
                "train_loss": loss.item(),
                "train_accuracy": accuracy
            })

            if i > 1:
                if i % 1000 == 0:
                    test(i, epoch)
                    model.train()
                if i % 500 == 0:
                    torch.save(model.state_dict(), f'e_{epoch}_i_{i}_bert_filter.pth')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_dataset = TextDataset(train_dataset_path)
    test_dataset = TextDataset(test_dataset_path)
    batch_size = 128
    epochs = 3

    wandb.init(
        project="bert_filter",
        config={
            "architecture": "bert",
            "epochs": epochs,
            "batch_size": batch_size
        }
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True)

    criterion = torch.nn.BCELoss()
    model = Model().to(device)
    train()
    wandb.finish()
