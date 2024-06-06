import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import Dataset

model_path = "bert-base-uncased"
token = BertTokenizer.from_pretrained(model_path)


class TextDataset(Dataset):
    def __init__(self, file_path, max_length=150):
        self.data = pd.read_csv(file_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = int(self.data.iloc[index]['label'])
        encoding = token.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    token_type_ids = pad_sequence([item['token_type_ids'] for item in batch], batch_first=True)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

    return input_ids, attention_mask, token_type_ids, labels
