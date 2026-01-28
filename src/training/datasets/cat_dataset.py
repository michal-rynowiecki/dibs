from transformers import AutoTokenizer
from torch import nn
import torch.optim as optim
import torch

import json

from torch.utils.data import Dataset, DataLoader, random_split

class CatDataset(Dataset):
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

    laptop_label2id_1 = {value: key for key, value in laptop_id2label_1.items()}
    laptop_label2id_2 = {value: key for key, value in laptop_id2label_2.items()}

    def __init__(self, data, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.inference = False
        self.ids = []

        for entry in data:
            id = entry['ID']
            # DANGER, here you are only keeping a single data example from each id
            #if id in self.ids:
            #    continue
            text = entry['Text']
            aspect = entry['Aspect']
            opinion = entry['Opinion']
            if 'Cat1' in entry:
                cat1 = self.laptop_label2id_1[entry['Cat1']] # Change the text label into id for putting into tensors later on
                cat2 = self.laptop_label2id_2[entry['Cat2']] # ^
                self.samples.append((id, text, aspect, opinion, cat1, cat2))

            else:
                self.inference = True
                self.samples.append((id, text, aspect, opinion))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.inference:
            id, text, aspect, opinion= self.samples[index]
        else:
            id, text, aspect, opinion, cat1, cat2 = self.samples[index]
        
        encoding = self.tokenizer(
            f"{aspect}[SEP]{opinion}",
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        if self.inference:
            return {
                'ID': id,
                'text': text,
                'aspect': aspect,
                'opinion': opinion,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
            }
        else:
            return {
                'ID': id,
                'text': text,
                'aspect': aspect,
                'opinion': opinion,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'cat1': torch.tensor(cat1, dtype=torch.long), # Use long for CrossEntropy
                'cat2': torch.tensor(cat2, dtype=torch.long) # Use long for CrossEntropy
            }