from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

from torchcrf import CRF

from read_input import read_data

class AspectModel(nn.Module):
    def __init__(self, model_path, num_tags):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

        hidden_size = self.bert.config.hidden_size

        # BERT hidden states to emission scores
        self.emission_layer = nn.Linear(hidden_size, num_tags)

        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        emissions = self.emission_layer(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss

        tag_sequence = self.crf.decode(emissions, mask=attention_mask.bool())
        return tag_sequence

if __name__=="__main__":

    train_path = "/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl"
    model_path = "prajjwal1/bert-tiny"

    tags = ["O", "B-Asp", "I-Asp"]
    num_tags = len(tags)

    model = AspectModel(model_path, num_tags)

    data = read_data(train_path)

    ID = "rest16_quad_dev_2"
    text = data[ID]["Text"]
    quad = data[ID]["Quadruplet"]

    tok = AutoTokenizer.from_pretrained(model_path)

    tokenized_text = tok(text, return_tensors="pt")
    
    with torch.no_grad():
        pred_tags = model(tokenized_text['input_ids'], tokenized_text['attention_mask'])
    
    print('Input tokens shape: ', tokenized_text['input_ids'].shape)
    print('Tags shape: ', len(pred_tags[0]))