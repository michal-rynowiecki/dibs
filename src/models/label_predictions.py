from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch import nn
from datasets import Dataset

from read_input import read_data

# Create a class that will add a classification head to any existing BERT model
class ExtendedModel(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
        # Get the hidden size to know the shape of the input for the head
        hidden_size = self.bert.config.hidden_size

        # Add the clasifier
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    # Use the cls token from the original model as the input to the linear layer
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


if __name__ == '__main__':

    train_path = "/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl"
    model_path = "prajjwal1/bert-tiny"

    data = read_data(train_path)

    entity2id = {
        'Restaurant': {"RESTAURANT": 0, "FOOD": 1, "DRINKS": 2, "AMBIENCE": 3,"SERVICE": 4, "LOCATION": 5},
        'Laptop': ["LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISK", "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING", "SUPPORT", "COMPANY", "OUT_OF_SCOPE"],
        'Hotel': ["HOTEL", "ROOMS" "FACILITIES", "ROOM_AMENITIES", "SERVICE", "LOCATION", "FOOD_DRINKS"],
        'Finance': ["MARKET", "COMPANY", "BUSINESS", "PRODUCT"]
    }

    # Retrieve a single data point
    ID = "rest16_quad_dev_2"
    text = data[ID]["Text"]
    quad = data[ID]["Quadruplet"]

    print(text)
    for elem in quad:
        print(elem)
    '''
    # Load in the tokenizer
    tok = AutoTokenizer.from_pretrained(model_path)

    # Tokenizer the singe data point
    tokenized_text = tok(text, return_tensors="pt")
    print(tokenized_text['input_ids'].shape)
    
    # Load in the model
    model = ExtendedModel(model_path, 2)  
    
    # Run the data through the model
    with torch.no_grad():
        outputs = model(tokenized_text['input_ids'], tokenized_text['attention_mask'])
    '''