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
    PATH = "/Users/michal/Projects/sentiment"

    path = f'{PATH}/data/processed/bin_restaurant_test_alltasks.jsonl'
    #model_path = "prajjwal1/bert-medium"
    model_path = "microsoft/deberta-v3-base"
    
    f_in = open(path, 'r')
    data = []
    for line in f_in.readlines():
        data.append(json.loads(line))
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = BinModel(model_path)

    test_size = 1

    
    dataset = BinDataset(data, tokenizer) #add_prefix_space=True

    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model.to(device)
    
    '''
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    
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
            print('Loss: ', loss)
            # Backward pass (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), f"{PATH}/src/models/bert-base/bin_model_stepone.pt")
    print("Model saved!")
    '''
    state_dict = torch.load(f"{PATH}/src/models/bert-base/bin_model_stepone.pt", map_location=torch.device('cpu'))
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    #f = open(f'{PATH}/data/predictions/test/eng_laptop_preds_bin_test.jsonl', 'w')
    with open(f'{PATH}/data/predictions/test/eng_restaurant_preds_bin_test.jsonl', "w") as f:
        for batch in test_loader:
            with torch.no_grad():

                ids = batch['id']['ID']
                sentences = batch['sentence']
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
                print(len(ids), len(aspects), len(opinions), len(sentences), len(pred_labels))
                
                for d in batch_output:
                    print(d)
                    json.dump(d, f)
                    f.write("\n")