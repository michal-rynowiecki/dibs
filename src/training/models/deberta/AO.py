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

import json

class CrossAttentionLayer(nn.Module):
    # Note: Fixed the 'droput' typo to 'dropout'
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None: 
            # PyTorch MultiheadAttention expects True for tokens that should be ignored (padding)
            mask = (mask == 0)
        
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=mask
        )
    
        output = self.norm(query + self.dropout(attn_output))
        return output

class DualModule(nn.Module):
    # Removed attn_layers argument since we only attend at the very end now
    def __init__(self, model1_name, model2_name, num_tags):
        super().__init__()

        # Renamed from bert1 to encoder1 for clarity, as DeBERTa is not strictly BERT
        self.encoder1 = AutoModel.from_pretrained(model1_name)
        self.encoder2 = AutoModel.from_pretrained(model2_name)

        hidden_size = self.encoder1.config.hidden_size

        # Initialize cross-attention for the final layer outputs
        self.cross_attn1 = CrossAttentionLayer(hidden_size)
        self.cross_attn2 = CrossAttentionLayer(hidden_size)

        self.emission1 = nn.Linear(hidden_size, num_tags)
        self.emission2 = nn.Linear(hidden_size, num_tags)
        
        self.crf1 = CRF(num_tags, batch_first=True)
        self.crf2 = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels1=None, labels2=None):
        # 1. Let DeBERTa handle the forward pass natively. 
        # This safely manages DeBERTa v3's disentangled attention and relative position biases.
        outputs1 = self.encoder1(input_ids, attention_mask=attention_mask, return_dict=True)
        outputs2 = self.encoder2(input_ids, attention_mask=attention_mask, return_dict=True)

        # 2. Extract the final hidden states (the output of the last layer)
        hidden1 = outputs1.last_hidden_state
        hidden2 = outputs2.last_hidden_state

        # 3. Apply Cross-Attention
        # Hidden 1 queries Hidden 2
        attended_hidden1 = self.cross_attn1(query=hidden1, key=hidden2, value=hidden2, mask=attention_mask)
        # Hidden 2 queries Hidden 1
        attended_hidden2 = self.cross_attn2(query=hidden2, key=hidden1, value=hidden1, mask=attention_mask)

        # 4. Get emissions
        emissions1 = self.emission1(attended_hidden1)
        emissions2 = self.emission2(attended_hidden2)

        # 5. Calculate Loss or Decode Tags
        if labels1 is not None and labels2 is not None:
            loss1 = -self.crf1(emissions1, labels1, mask=attention_mask.bool(), reduction='mean')
            loss2 = -self.crf2(emissions2, labels2, mask=attention_mask.bool(), reduction='mean')
            return loss1 + loss2
        
        else:
            tags1 = self.crf1.decode(emissions1, mask=attention_mask.bool())
            tags2 = self.crf2.decode(emissions2, mask=attention_mask.bool())
            return tags1, tags2