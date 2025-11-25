from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch
from torch import nn

from read_input import read_data

if __name__=="__main__":
    
    model_path = "KoichiYasuoka/roberta-large-english-upos"
    train_path = "/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl"

    model = AutoModel.from_pretrained(model_path)

    data = read_data(train_path)

    ID = "rest16_quad_dev_2"
    text = data[ID]["Text"]
    quad = data[ID]["Quadruplet"]

    tok = AutoTokenizer.from_pretrained(model_path)

    tokenized_text = tok(text, return_tensors="pt")

    with torch.no_grad():
        context_embedings = model(**tokenized_text)
