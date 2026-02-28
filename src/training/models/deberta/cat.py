import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from torch.utils.data import Dataset, DataLoader, random_split

from utils.read_input import read_data, read_conll
from utils.transform_tokens import get_entities, get_entities_batch

class CrossAttentionLayer(nn.Module):
    # Fixed the 'droput' typo
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None: 
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
    def __init__(self, model1_name, model2_name, classes1, classes2, dropout_rate=0.1, class1_weights=None, class2_weights=None):
        super().__init__()

        # Renamed to encoder1 and encoder2 for clarity
        self.encoder1 = AutoModel.from_pretrained(model1_name)
        self.encoder2 = AutoModel.from_pretrained(model2_name)

        # Gentle correction: Use 1s instead of 0s for default weights so the loss doesn't zero out
        if class1_weights is None:
            self.class1_weights = torch.ones(len(classes1))
            self.class2_weights = torch.ones(len(classes2))
        else:
            self.class1_weights = class1_weights
            self.class2_weights = class2_weights

        hidden_size = self.encoder1.config.hidden_size

        # Initialize cross-attention for the final layer outputs only
        self.cross_attn1 = CrossAttentionLayer(hidden_size)
        self.cross_attn2 = CrossAttentionLayer(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.classes1 = classes1
        self.classes2 = classes2

        # Pooler layers for CLS token extraction
        self.pooler1 = nn.Linear(hidden_size, hidden_size)
        self.pooler2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        # Classification heads
        self.classifier1 = nn.Linear(hidden_size, len(classes1))
        self.classifier2 = nn.Linear(hidden_size, len(classes2))

        self.loss_fct1 = nn.CrossEntropyLoss(weight=self.class1_weights)
        self.loss_fct2 = nn.CrossEntropyLoss(weight=self.class2_weights)

    def forward(self, input_ids, attention_mask=None, labels1=None, labels2=None):
        # 1. Standard forward pass through DeBERTa
        outputs1 = self.encoder1(input_ids, attention_mask=attention_mask)
        outputs2 = self.encoder2(input_ids, attention_mask=attention_mask)

        # 2. Extract the final hidden states
        hidden1 = outputs1.last_hidden_state
        hidden2 = outputs2.last_hidden_state

        # 3. Apply Cross-Attention
        attended_hidden1 = self.cross_attn1(query=hidden1, key=hidden2, value=hidden2, mask=attention_mask)
        attended_hidden2 = self.cross_attn2(query=hidden2, key=hidden1, value=hidden1, mask=attention_mask)

        # 4. Extract CLS tokens (the first token of the sequence) from the cross-attended representations
        cls_token1 = attended_hidden1[:, 0, :]
        cls_token2 = attended_hidden2[:, 0, :]

        # 5. Apply Pooler Logic
        pooled_output1 = self.tanh(self.pooler1(cls_token1))
        pooled_output2 = self.tanh(self.pooler2(cls_token2))

        # 6. Apply Dropout
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)

        # 7. Get Logits
        logits1 = self.classifier1(pooled_output1)
        logits2 = self.classifier2(pooled_output2)

        # 8. Calculate Loss or Return Logits
        if labels1 is not None and labels2 is not None:
            # Note: Ensure self.class1_weights are on the same device as logits1 in your training loop!
            loss1 = self.loss_fct1(logits1, labels1)
            loss2 = self.loss_fct2(logits2, labels2)
            return 0.7 * loss1 + loss2
        
        else:
            return logits1, logits2