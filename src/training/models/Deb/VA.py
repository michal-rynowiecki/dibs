import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None: 
            # MultiheadAttention requires True for padded positions
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
    def __init__(self, model1_name, model2_name, dropout_rate=0.1):
        super().__init__()

        # Load DeBERTa backbones
        self.encoder1 = AutoModel.from_pretrained(model1_name)
        self.encoder2 = AutoModel.from_pretrained(model2_name)

        hidden_size = self.encoder1.config.hidden_size

        # Cross-attention layers (applied only after the final layer)
        self.cross_attn1 = CrossAttentionLayer(hidden_size)
        self.cross_attn2 = CrossAttentionLayer(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        # Regression heads: Linear(hidden_size -> 1)
        self.regressor1 = nn.Linear(hidden_size, 1)
        self.regressor2 = nn.Linear(hidden_size, 1)

        # Regression losses
        self.loss_v = nn.MSELoss()
        self.loss_a = nn.MSELoss()

    def forward(self, input_ids, attention_mask=None, gold1=None, gold2=None):
        # 1. Forward pass through DeBERTa backbones
        # DeBERTa v3 handles its own relative position embeddings and disentangled attention
        outputs1 = self.encoder1(input_ids, attention_mask=attention_mask)
        outputs2 = self.encoder2(input_ids, attention_mask=attention_mask)

        # 2. Extract final hidden states
        hidden1 = outputs1.last_hidden_state
        hidden2 = outputs2.last_hidden_state

        # 3. Apply Cross-Attention logic
        # Model 1 attends to context from Model 2 and vice-versa
        attended_hidden1 = self.cross_attn1(query=hidden1, key=hidden2, value=hidden2, mask=attention_mask)
        attended_hidden2 = self.cross_attn2(query=hidden2, key=hidden1, value=hidden1, mask=attention_mask)

        # 4. Extract CLS tokens for regression
        # In DeBERTa, index 0 is used for the representation of the whole sequence
        cls_embedding1 = self.dropout(attended_hidden1[:, 0, :])
        cls_embedding2 = self.dropout(attended_hidden2[:, 0, :])

        # 5. Calculate regression scores
        score1 = self.regressor1(cls_embedding1)
        score2 = self.regressor2(cls_embedding2)

        if gold1 is not None and gold2 is not None:
            # Flatten to match shape (batch_size,) if gold labels are 1D
            loss1 = self.loss_v(score1.view(-1), gold1.view(-1))
            loss2 = self.loss_a(score2.view(-1), gold2.view(-1))
            return loss1 + loss2
        
        else:
            return score1, score2