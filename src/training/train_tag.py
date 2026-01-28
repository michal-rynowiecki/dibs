from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

import torch.optim as optim
from torch.nn import CrossEntropyLoss

from torchcrf import CRF

from utils.read_input import read_data, read_conll
from utils.transform_tokens import align_labels_with_tokens
from BIO_dataset import BIODataset
from tag import TagModule

from torch.utils.data import Dataset, DataLoader, random_split


def map_tokens_to_words(tokenizer, input_ids, gold_labels1, pred_labels1, gold_labels2=None, pred_labels2=None):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(sentence)
    words = words[:len(pred_labels1)]
    if pred_labels2:
        print("Word \t Prediction1 \t Gold Label \t Prediction2 \t Gold Label2")
        for index, word in enumerate(words):
            print(f"{word} \t\t {pred_labels1[index]} \t\t {gold_labels1[index]} \t\t{pred_labels2[index]} \t\t {gold_labels2[index]}")
    else:
        print("Word \t Prediction1 \t Gold Label")
        for index, word in enumerate(words):
            print(f"{word} \t\t {pred_labels1[index]} \t\t {gold_labels1[index]}")

if __name__== "__main__":

    data_path = "/Users/michal/Projects/sentiment/data/tagged/eng_laptop_train_BIO_Opinion.jsonl"
    model_path = "prajjwal1/bert-small"

    # Tag initialization
    tag_to_id = {"O": 0, "B-Asp": 1, "I-Asp": 2}
    id_to_tag = {0: "O",1: "B-Asp",2: "I-Asp"}

    # Read in the model and data
    model = TagModule(model_path, len(tag_to_id))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = read_conll(data_path)
    
    
    # Put data in the data loader
    train_size = 0.8
    test_size = 0.2

    dataset = BIODataset(data, tokenizer, tag_to_id)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
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
            labels = batch['labels'].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model(input_ids, attention_mask, labels)

            # Backward pass (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "/Users/michal/Projects/sentiment/src/models/opinion_model_weights.pt")
    print("Model saved!")