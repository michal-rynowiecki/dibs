from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch
from torch import nn

from read_input import read_data

class ValenceModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

        hidden_size = self.bert.config.hidden_size

        self.regression = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:,0,:]
        score = self.regression(cls_embedding)
        return score

if __name__=="__main__":

    train_path = "/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl"
    model_path = "prajjwal1/bert-tiny"

    data = read_data(train_path)

    ID = "rest16_quad_dev_2"
    text = data[ID]["Text"]
    quad = data[ID]["Quadruplet"]

    input = "Aspect : sake list , Opinion : extensive , Category : DRINKS , STYLE OPTIONS , Text : their sake list was extensive , but we were looking for purple haze , which was n ' t listed but made for us upon request !"

    tok = AutoTokenizer.from_pretrained(model_path)

    tokenized_text = tok(input, return_tensors="pt")
    print(tokenized_text)

    model = ValenceModel(model_path)

    with torch.no_grad():
        output = model(tokenized_text['input_ids'], tokenized_text['attention_mask'])