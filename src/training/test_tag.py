from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

from torchcrf import CRF

from utils.read_input import read_data, read_conll
from utils.transform_tokens import align_labels_with_tokens
from BIO_dataset import BIODataset

from torch.utils.data import Dataset, DataLoader, random_split
from tag import TagModule
from train_tag import map_tokens_to_words


def calculate_accuracy(predictions1, labels1, attention_mask):
    """
    Calculates accuracy for a batch of lists with varying lengths.
    Inputs are expected to be lists of lists.
    """
    total_correct = 0
    total_tokens = 0

    # Iterate through each sample in the batch
    for p1, l1, mask in zip(predictions1, labels1, attention_mask):
        print('Predictions 1: ', p1)
        print('Labels 1: ', l1)
        
        # Iterate through each token in the sequence
        # We use l1 to determine the range to ensure we stay within bounds
        for i in range(len(l1)):
            if mask[i] == 1:
                # Check mask (not padding) and label (not 'O' or 0)
                if l1[i] != 0:
                    total_tokens += 1
                    if p1[i] == l1[i]:
                        total_correct += 1

    if total_tokens == 0:
        return 0, 0

    return total_correct, total_tokens

if __name__ == "__main__":    
    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cpu")
    data_path = "/Users/michal/Projects/sentiment/data/tagged/eng_laptop_dev_BIO_opinion.jsonl"
    model_path = "prajjwal1/bert-small"

    # Tag initialization
    tag_to_id = {"O": 0, "B-Asp": 1, "I-Asp": 2}
    id_to_tag = {0: "O",1: "B-Asp",2: "I-Asp"}

    # Read in the model and data
    model = TagModule(model_path, len(tag_to_id))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = read_conll(data_path)

    # Put data in the data loader
    test_size = 0.2

    dataset = BIODataset(data, tokenizer, tag_to_id)
    _, test_dataset = random_split(dataset, [1-test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    state_dict = torch.load("/Users/michal/Projects/sentiment/src/models/opinion_model_weights.pt", map_location=torch.device('mps'))

    model.load_state_dict(state_dict)
    
    total_correct = 0
    total_tokens = 0

    
    # Testing
    for batch in test_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask)

            map_tokens_to_words(tokenizer, input_ids[0], labels[0], predictions[0])

            correct, count = calculate_accuracy(predictions, labels, attention_mask)
            total_correct += correct
            total_tokens += count

    test_accuracy = total_correct / total_tokens
    print(f"Test Accuracy: {test_accuracy:.4f}")