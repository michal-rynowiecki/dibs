import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from torch.utils.data import Dataset, DataLoader, random_split

from utils.read_input import read_data, read_conll
from utils.transform_tokens import get_entities, get_entities_batch

from datasets.cat_dataset import CatDataset

from train_tag import map_tokens_to_words

from models.cat import DualModule
from collections import Counter

from random import shuffle

import json

def combine(quadruplet, line):
    # Retrieve the aspect and the opinion from the quadruplet
    aspect = quadruplet['Aspect']
    opinion = quadruplet['Opinion']
    input = f"{aspect}[SEP]{opinion}[SEP]{line}"

    return input

def get_damped_class_weights(counts, device):
    """
    Calculates weights using Square Root Inverse Frequency.
    counts: List or array of sample counts per class.
    """
    counts = np.array(counts)
    # Avoid division by zero
    counts[counts == 0] = 1 
    
    # Calculate max freq to normalize (Majority class gets weight ~1.0)
    max_count = np.max(counts)
    
    # Formula: sqrt(Max / Count)
    weights = np.sqrt(max_count / counts)
    
    # Convert to Tensor
    return torch.tensor(weights, dtype=torch.float32).to(device)


if __name__ == "__main__":
    INFERENCE = False
    PATH = "/Users/michal/Projects/sentiment"
    laptop_id2label_1 = {
        0: 'LAPTOP',
        1: 'HARDWARE', 
        2: 'DISPLAY',
        3: 'KEYBOARD', 
        4: 'BATTERY',
        5: 'SOFTWARE',
        6: 'SUPPORT',
        7: 'MULTIMEDIA_DEVICES', 
        8: 'OS',
        9: 'HARD_DISK',
        10: 'FANS_COOLING',
        11: 'COMPANY',
        12: 'CPU',
        13: 'MEMORY',
        14: 'OPTICAL_DRIVES',
        15: 'PORTS',
        16: 'GRAPHICS',
        17: 'SHIPPING',
        18: 'POWER_SUPPLY',
        19: 'MOUSE',
        20: 'WARRANTY',
        21: 'MOTHERBOARD',
        22: 'OUT_OF_SCOPE'
    }

    laptop_id2label_2 = {
        0: 'DESIGN_FEATURES',
        1: 'GENERAL',
        2: 'OPERATION_PERFORMANCE',
        3: 'QUALITY',
        4: 'USABILITY',
        5: 'PRICE',
        6: 'PORTABILITY',
        7: 'MISCELLANEOUS',
        8: 'CONNECTIVITY'
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model_path = "prajjwal1/bert-mini"
    model_path = "FacebookAI/roberta-large"
    data_path = f"{PATH}/data/processed/categories_eng_laptop_train_alltasks.jsonl"
    #data_path = f"{PATH}/data/predictions/eng_laptop_preds_bin.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    f = open(data_path, 'r')

    data = []
    if not INFERENCE:
        cat1_counts = {a: 0 for a in laptop_id2label_1.values()}
        cat2_counts = {a: 0 for a in laptop_id2label_2.values()}

        for line in f.readlines():
            temp = json.loads(line)
            text = temp['Text']
            id = temp['ID']

            for elem in temp['Quadruplet']:
                input = combine(elem, text) # combines the relevant elements and the data point text into model input (pre-tokenizer)
                elem['Input'] = input

                cat1_counts[elem['Cat1']] += 1
                cat2_counts[elem['Cat2']] += 1

                data.append({'ID': id, 
                'Text': text, 
                'Aspect': elem['Aspect'], 
                'Opinion': elem['Opinion'],
                'Cat1': elem['Cat1'],
                'Cat2': elem['Cat2']
                })
    else:
        for line in f.readlines():
            temp = json.loads(line)

            text = temp['sentence']
            id = temp['id']

            data.append({'ID': id, 
                'Text': text, 
                'Aspect': temp['aspect'], 
                'Opinion': temp['opinion'],
                })

    if not INFERENCE:
        # Get class weights
        class_weights1 = get_damped_class_weights(list(cat1_counts.values()), device)
        class_weights2 = get_damped_class_weights(list(cat2_counts.values()), device)

        # Remove this if you remove the only single data point per item
        shuffle(data)

    dataset = CatDataset(data, tokenizer)

    test_size= 0
    
    train_dataset, test_dataset = random_split(dataset, [1-test_size,  test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    if not INFERENCE:
        model = DualModule(model_path, model_path, list(laptop_id2label_1.values()), list(laptop_id2label_2.values()), attn_layers=[7, 12, 23], class1_weights=class_weights1, class2_weights=class_weights2)
    else:
        model = DualModule(model_path, model_path, list(laptop_id2label_1.values()), list(laptop_id2label_2.values()), attn_layers=[7, 12, 23])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    epochs = 3
    model.train()
    
    for epoch in range(epochs):
        
        total_loss = 0

        for batch in train_loader:
            print(batch)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels1 = batch['cat1'].to(device)
            labels2 = batch['cat2'].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model(input_ids, attention_mask=attention_mask, labels1=labels1, labels2=labels2)

            # Backward pass (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), f"{PATH}/src/models/cat_model_stepone.pt")
    print("Model saved!")
    '''
    state_dict = torch.load(f"{PATH}/src/models/cat_model.pt", map_location=torch.device('mps'))
    model.load_state_dict(state_dict)

    f = open(f"{PATH}/data/predictions/eng_laptop_preds_cat.jsonl", 'w')
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            texts = batch['text']
            aspect = batch['aspect']
            opinion = batch['opinion']
            attention_mask = batch['attention_mask'].to(device)
            if not INFERENCE:
                labels1 = batch['cat1'].to(device)
                labels2 = batch['cat2'].to(device)
            ids = batch['ID']

            predictions1, predictions2 = model(input_ids, attention_mask=attention_mask)
            predictions1 = predictions1.argmax(dim=1)
            predictions2 = predictions2.argmax(dim=1)

            results = [
                {
                    "ID": id_, 
                    "Text": text,
                    "Aspect": asp,
                    "Opinion": op,
                    "Cat1": laptop_id2label_1[p1.item()],
                    "Cat2": laptop_id2label_2[p2.item()]
                }
                for id_,text, asp, op, p1, p2 in zip(ids, texts, aspect, opinion, predictions1, predictions2)
            ]
            for result in results:
                json.dump(result, f)
                f.write("\n")
            
    
    f = open(f"{PATH}/data/predictions/eng_laptop_preds_cat.jsonl", 'r')

    
    total = 0
    cor1 = 0
    cor2 = 0
    both = 0
    for line in f.readlines():
        cur = json.loads(line)
        cur_id = cur['id']
        for point in data:
            if point['ID'] == cur_id and point['Aspect'] == cur['aspect'] and point['Opinion'] == cur['opinion']:
                if cur['prediction1'] == point['Cat1']:
                    cor1 += 1
                if cur['prediction2'] == point['Cat2']:
                    cor2 += 1
                if cur['prediction1'] == point['Cat1'] and cur['prediction2'] == point['Cat2']:
                    both += 1 
                total +=1

    print('Total: ', total)
    print('Category 1 correct: ', cor1/total)
    print('Category 2 correct: ', cor2/total)
    print('Both: ', both/total)
    '''