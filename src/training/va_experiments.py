import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from torch.utils.data import Dataset, DataLoader, random_split

from utils.read_input import read_data, read_conll
from utils.transform_tokens import get_entities, get_entities_batch

from datasets.va_dataset import VADataset

from train_tag import map_tokens_to_words

from models.VA import DualModule
from collections import Counter

import json

if __name__ == "__main__":
    INFERENCE = False
    PATH = "/Users/michal/Projects/sentiment"
    
    path_data   = "/Users/michal/Projects/sentiment/data/processed/va_eng_laptop_train_alltasks.jsonl" # Path to the OG dataset which also contains the text data (i.e. actual sentences)
    #path_data   = f"{PATH}/data/predictions/eng_laptop_preds_cat.jsonl"
    model_path  = "prajjwal1/bert-medium"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    f = open(path_data, 'r')

    data = []
    if not INFERENCE:
        for line in f.readlines():
            temp = json.loads(line)
            text = temp['Text']
            id = temp['ID']

            for elem in temp['Quadruplet']:
                data.append({'ID': id,
                'Text': text, 
                'Aspect': elem['Aspect'], 
                'Opinion': elem['Opinion'],
                'Cat1': elem['Cat1'], 
                "Cat2": elem['Cat2'], 
                "Valence": elem['Valence'], 
                "Arousal": elem['Arousal']
                })

    # Else the inference time read in (each line has only a single prediction)
    else:
        for line in f.readlines():
            temp = json.loads(line)
            
            data.append({
            'Text': temp['Text'],
            'ID': temp['ID'],
            'Aspect': temp['Aspect'],
            'Opinion': temp['Opinion'],
            'Cat1': temp['Cat1'],
            "Cat2": temp['Cat2'], 
            })

    dataset = VADataset(data, tokenizer)

    test_size=0
    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = DualModule(model_path, model_path, attn_layers=[4, 7])

    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            valence = batch['valence'].to(device)
            arousal = batch['arousal'].to(device)

            optimizer.zero_grad()

            loss = model(input_ids, attention_mask=attention_mask, gold1=valence, gold2=arousal)
            print('Loss: ', loss)
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"{PATH}/src/models/bert-base/va_model_laptop.pt")
    print("Model saved!")
    '''
    state_dict = torch.load("f"{PATH}/src/models/va_model.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    f = open(f"{PATH}/data/predictions/eng_laptop_preds_va.jsonl", 'w')

    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            aspect = batch['aspect']
            opinion = batch['opinion']
            attention_mask = batch['attention_mask'].to(device)
            if not INFERENCE:
                valence = batch['valence']
                arousal = batch['arousal'] 
            texts = batch['text']           
            ids = batch['ID']
            cats1 = batch['cat1']
            cats2 = batch['cat2']

            v_preds, a_preds = model(input_ids, attention_mask=attention_mask)
            print(ids)
            results = [
                {
                    "id": id_,
                    "text": text,
                    "aspect": asp,
                    "opinion": op,
                    "cat1": cat1,
                    "cat2": cat2,
                    "valence": v.item(),
                    "arousal": a.item()
                }
                for id_,text,asp, op,cat1,cat2, v, a in zip(ids,texts, aspect, opinion,cats1,cats2, v_preds, a_preds)
            ]
            
            for result in results:
                json.dump(result, f)
                f.write("\n")
    '''