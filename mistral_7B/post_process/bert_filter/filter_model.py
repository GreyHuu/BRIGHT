import torch
from transformers import BertModel, BertTokenizer

model_path = "bert-base-uncased"


# model 4
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.drop = torch.nn.Dropout(0.3)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        outputs = out.last_hidden_state[:, 0, :]
        outputs = self.drop(outputs)
        return self.predictor(outputs)


