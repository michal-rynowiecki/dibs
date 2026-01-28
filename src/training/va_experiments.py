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
    path_data   = "/Users/michal/Projects/sentiment/data/processed/va_eng_laptop_train_alltasks.jsonl" # Path to the OG dataset which also contains the text data (i.e. actual sentences)
    model_path  = "prajjwal1/bert-tiny"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    f = open(path_data, 'r')

    data = []
    
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

    dataset = VADataset(data, tokenizer)

    test_size=0.2
    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = DualModule(model_path, model_path)

    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    '''
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
            
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "/Users/michal/Projects/sentiment/src/models/va_model.pt")
    print("Model saved!")
    '''
    state_dict = torch.load("/Users/michal/Projects/sentiment/src/models/va_model.pt", map_location=torch.device('mps'))
    model.load_state_dict(state_dict)

    f = open("/Users/michal/Projects/sentiment/data/predictions/eng_laptop_preds_va.jsonl", 'w')

    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            aspect = batch['aspect']
            opinion = batch['opinion']
            attention_mask = batch['attention_mask'].to(device)
            valence = batch['valence']
            arousal = batch['arousal']            
            ids = batch['ID']

            v_preds, a_preds = model(input_ids, attention_mask=attention_mask)

            results = [
                {
                    "id": id_, 
                    "aspect": asp,
                    "opinion": op,
                    "valence": v.item(),
                    "arousal": a.item()
                }
                for id_,asp, op, v, a in zip(ids, aspect, opinion, v_preds, a_preds)
            ]
            print(results)
            for result in results:
                json.dump(result, f)
                f.write("\n")
    

