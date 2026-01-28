import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils.read_input import read_conll

class BIODataset(Dataset):
    def __init__(self, data, tokenizer, tag_to_id, max_len=254):
        self.data = data
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract words and string tags
        id = self.data[idx]['ID']
        words = [pair[0] for pair in self.data[idx]['Tags']]
        tags = [pair[1] for pair in self.data[idx]['Tags']]
        
        # 1. Tokenize
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length', # Ensures all sentences in a batch are same length
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # 2. Align labels
        word_ids = encoding.word_ids()
        label_ids = [self.tag_to_id[t] for t in tags]
        
        aligned_label_ids = []
        current_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_label_ids.append(0)
            elif word_idx != current_word_idx:
                # First subword: get actual tag
                aligned_label_ids.append(label_ids[word_idx])
            else:
                # Subsequent subwords
                aligned_label_ids.append(label_ids[word_idx])
                #aligned_label_ids.append(0)
            current_word_idx = word_idx

        # Remove the extra batch dimension added by return_tensors="pt" 
        # because the DataLoader will add its own batch dimension
        return {
            'ID': id,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_label_ids)
        }

class BIODatasetDouble(Dataset):
    def __init__(self, data1, data2, tokenizer, tag_to_id, max_len=128):
        self.data1 = data1
        self.data2 = data2
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        # Extract words and string tags
        id = self.data1[idx]['ID']
        words = [pair[0] for pair in self.data1[idx]['Tags']]
        tags1 = [pair[1] for pair in self.data1[idx]['Tags']]
        tags2 = [pair[1] for pair in self.data2[idx]['Tags']]
        
        # 1. Tokenize
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length', # Ensures all sentences in a batch are same length
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        print(encoding)

        # 2. Align labels
        word_ids = encoding.word_ids()
        print(word_ids)
        label_ids1 = [self.tag_to_id[t1] for t1 in tags1]
        label_ids2 = [self.tag_to_id[t2] for t2 in tags2]
        
        aligned_label_ids1 = []
        aligned_label_ids2 = []
        current_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_label_ids1.append(0)
                aligned_label_ids2.append(0)
            elif word_idx != current_word_idx:
                # First subword: get actual tag
                aligned_label_ids1.append(label_ids1[word_idx])
                aligned_label_ids2.append(label_ids2[word_idx])
            else:
                # Subsequent subwords
                aligned_label_ids1.append(label_ids1[word_idx])
                aligned_label_ids2.append(label_ids2[word_idx])
                #aligned_label_ids1.append(0)
                #aligned_label_ids2.append(0)
            current_word_idx = word_idx

        # Remove the extra batch dimension added by return_tensors="pt" 
        # because the DataLoader will add its own batch dimension
        return {
            'ID': id,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels1': torch.tensor(aligned_label_ids1),
            'labels2': torch.tensor(aligned_label_ids2)
        }

class BIODatasetDouble_new(Dataset):
    def __init__(self, data1, data2, tokenizer, tag_to_id, max_len=128):
        self.data1 = data1
        self.data2 = data2
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        # Extract words and string tags
        id = self.data1[idx]['ID']
        words = [pair[0] for pair in self.data1[idx]['Tags']]
        tags1 = [pair[1] for pair in self.data1[idx]['Tags']]
        tags2 = [pair[1] for pair in self.data2[idx]['Tags']]

        print(words)
        print(tags1)
        print(tags2)
    
        # 1. Tokenize
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length', # Ensures all sentences in a batch are same length
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        print(encoding['input_ids'])
        word_ids = encoding.word_ids()
        print(word_ids)
