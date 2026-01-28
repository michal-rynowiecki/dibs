from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch.optim as optim
import torch
import itertools

import random

from models.binary import BinModel
from datasets.bin_dataset import BinDataset

import json

from utils.read_input import read_data

from torch.utils.data import Dataset, DataLoader, random_split

if __name__ == "__main__":
    INFERENCE = True

    path = '/Users/michal/Projects/sentiment/data/processed/bin_laptop_train_alltasks.jsonl'
    model_path = "prajjwal1/bert-tiny"
    
    f_in = open(path, 'r')
    data = []
    for line in f_in.readlines():
        data.append(json.loads(line))
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = BinModel(model_path)

    test_size = 0.2

    dataset = BinDataset(data, tokenizer)

    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    '''
    epochs = 2
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)['loss']

            # Backward pass (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "/Users/michal/Projects/sentiment/src/models/bin_model.pt")
    print("Model saved!")
    '''
    state_dict = torch.load("/Users/michal/Projects/sentiment/src/models/bin_model.pt", map_location=torch.device('mps'))
    model.load_state_dict(state_dict)

    f = open('/Users/michal/Projects/sentiment/data/predictions/eng_laptop_preds_bin.jsonl', 'w')

    for batch in test_loader:
        with torch.no_grad():

            ids = batch['id']['ID']
            aspects = batch['aspect']
            opinions = batch['opinion']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            if not INFERENCE:
                labels = batch['labels'].to(device)

            preds = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            probs = torch.sigmoid(preds) 
            pred_labels = (probs > 0.5).long()

            sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            if not INFERENCE:
                batch_output = [
                {
                    "id": id,
                    "aspect": aspect,
                    "opinion": opinion,
                    "sentence": sent,
                    "predicted label": pred_label.item(),
                    "gold label": gold.item()
                } for id, aspect, opinion, sent, pred_label, gold in zip(
                    ids, aspects, opinions, sentences, pred_labels, labels
                ) #if pred_label.item() == gold.item() and pred_label.item() == 1
                if pred_label.item() == 1]

            else:
                batch_output = [
                {
                    "id": id,
                    "aspect": aspect,
                    "opinion": opinion,
                    "sentence": sent,
                } for id, aspect, opinion, sent, pred_label in zip(
                    ids, aspects, opinions, sentences, pred_labels
                ) #if pred_label.item() == gold.item() and pred_label.item() == 1
                if pred_label.item() == 1]              

            for d in batch_output:
                json.dump(d, f)
                f.write("\n")