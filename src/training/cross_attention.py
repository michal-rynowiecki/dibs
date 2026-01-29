import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader, random_split

from utils.read_input import read_data, read_conll
from utils.transform_tokens import get_entities, get_entities_batch

from BIO_dataset import BIODataset, BIODatasetDouble

from train_tag import map_tokens_to_words

from models.AO import DualModule

import json
PATH = '/Users/michal/Projects/sentiment'
def calculate_accuracy(predictions1, predictions2, labels1, labels2, attention_mask):
    total_correct = 0
    total_tokens = 0

    # Iterate through each sample in the batch
    for p1, p2, l1, l2, mask in zip(predictions1, predictions2, labels1, labels2, attention_mask):
        
        # Iterate through each token in the sequence
        # We use l1 to determine the range to ensure we stay within bounds
        for i in range(len(l1)):
            if mask[i] == 1:
                # Check mask (not padding) and label (not 'O' or 0)
                if l1[i] != 0:
                    total_tokens += 1
                    if p1[i] == l1[i]:
                        total_correct += 1
                if l2[i] != 0:
                    total_tokens += 1
                    if p2[i] == l2[i]:
                        total_correct += 1

    if total_tokens == 0:
        return 0, 0

    return total_correct, total_tokens


if __name__== "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    data1_path = f"{PATH}/data/tagged/eng_laptop_restaurant_BIO_aspect.jsonl"
    #data1_path = '/Users/michal/Projects/sentiment/data/tagged/eng_laptop_dev_BIO_Aspect.jsonl'
    data2_path = f"{PATH}/data/tagged/eng_laptop_restaurant_BIO_Opinion.jsonl"
    #data2_path = '/Users/michal/Projects/sentiment/data/tagged/eng_laptop_dev_BIO_Opinion.jsonl'
    
    #model_path = "prajjwal1/bert-tiny"
    model_path = "FacebookAI/roberta-large"


    tag_to_id = {"O": 0, "B-Asp": 1, "I-Asp": 2}
    id_to_tag = {0: "O",1: "B-Asp",2: "I-Asp"}

    module = DualModule(model_path, model_path, 3)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data1 = read_conll(data1_path)
    data2 = read_conll(data2_path)

    # Load in the model trained on the previous dataset
    #state_dict = torch.load(f"{PATH}/src/models/Pipe/Asp_Op/aspect_opinion_model_weights_stepone.pt", map_location=torch.device('mps'))
    #module.load_state_dict(state_dict)

    test_size = 0

    dataset = BIODatasetDouble(data1, data2, tokenizer, tag_to_id)
    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    module.to(device)
    

    optimizer = optim.AdamW(module.parameters(), lr=5e-5)
    
    epochs = 3
    module.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels1 = batch['labels1'].to(device)
            labels2 = batch['labels2'].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            loss = module(input_ids, attention_mask=attention_mask, labels1=labels1, labels2=labels2)

            # Backward pass (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    torch.save(module.state_dict(), f"{PATH}/src/models/Pipe/Asp_Op/aspect_opinion_model_weights_stepone.pt")
    print("Model saved!")
    '''
    state_dict = torch.load(f'{PATH}/src/models/Pipe/Asp_Op/aspect_opinion_model_weights_stepone.pt')
    module.load_state_dict(state_dict)

    total_correct = 0
    total_tokens = 0
    
    f = open(f'{PATH}/data/predictions/eng_laptop_preds_double_train.jsonl', 'w')
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels1 = batch['labels1'].to(device)
            labels2 = batch['labels2'].to(device)
            ids = batch['ID']
            
            predictions1, predictions2 = module(input_ids, attention_mask=attention_mask)
            
            writing_dict = {}

            aspect_pred, aspect_label = get_entities_batch(tokenizer, input_ids, predictions1, labels1)
            opinion_pred, opinion_label = get_entities_batch(tokenizer, input_ids, predictions2, labels2)
            sentences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            batch_output = [
            {
                "ID": id,
                "sentence": sent,
                "aspect_predicted_tags": asp_t,
                "aspect_gold_labels": asp_g,
                "opinion_predicted_tags": op_t,
                "opinion_gold_labels": op_g
            }
            for id, sent, asp_t, asp_g, op_t, op_g  in zip(
                ids, sentences, aspect_pred, aspect_label, opinion_pred, opinion_label
            )]

            for d in batch_output:
                json.dump(d, f)
                f.write("\n")
            
            correct, count = calculate_accuracy(predictions1, predictions2, labels1, labels2, attention_mask)
            
            total_correct += correct
            total_tokens += count

    test_accuracy = total_correct / total_tokens
    print(f"Test Accuracy: {test_accuracy:.4f}")
    '''