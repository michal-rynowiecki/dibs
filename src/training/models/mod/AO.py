import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        # ModernBERT uses a specific epsilon for LayerNorm (1e-5 or 1e-6)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # PyTorch MultiheadAttention key_padding_mask: True means "ignore this token"
        if mask is not None: 
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None

        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=src_key_padding_mask
        )
    
        return self.norm(query + self.dropout(attn_output))

class DualModule(nn.Module):
    def __init__(self, model_id1, model_id2, num_tags, attn_layers=None):
        super().__init__()

        # Load ModernBERT models
        self.bert1 = AutoModel.from_pretrained(model_id1,attn_implementation="sdpa")
        self.bert2 = AutoModel.from_pretrained(model_id2,attn_implementation="sdpa")

        hidden_size = self.bert1.config.hidden_size
        # ModernBERT uses 'num_hidden_layers' in config
        num_layers = self.bert1.config.num_hidden_layers

        self.attn_indices = attn_layers if attn_layers is not None else [num_layers - 1]

        self.attn1_layers = nn.ModuleDict({
            f"layer_{i}": CrossAttentionLayer(hidden_size) for i in self.attn_indices
        })
        self.attn2_layers = nn.ModuleDict({
            f"layer_{i}": CrossAttentionLayer(hidden_size) for i in self.attn_indices
        })

        self.emission1 = nn.Linear(hidden_size, num_tags)
        self.emission2 = nn.Linear(hidden_size, num_tags)
        self.crf1 = CRF(num_tags, batch_first=True)
        self.crf2 = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels1=None, labels2=None):
        device = input_ids.device
        seq_length = input_ids.shape[1]
        # Create a range [0, 1, 2, ..., seq_length-1] for the batch
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if attention_mask is not None:
            processed_mask = attention_mask.to(device=device, dtype=torch.bool)
        else:
            processed_mask = None

        # 1. Get Initial Embeddings
        # ModernBERT has a 'embeddings' attribute but the logic inside the 
        # forward pass is cleaner if we use the layers as intended.
        h1 = self.bert1.embeddings(input_ids)
        h2 = self.bert2.embeddings(input_ids)

        # 2. Iterate through ModernBERT Layers
        # ModernBERT uses 'self.bert.layers' not 'self.bert.encoder.layer'
        for i, (layer1, layer2) in enumerate(zip(self.bert1.layers, self.bert2.layers)):
            # ModernBERT layers expect (hidden_states, attention_mask)
            # It handles the mask internally (often using Flash Attention)
            h1 = layer1(h1, processed_mask, position_ids=position_ids)[0]
            h2 = layer2(h2, processed_mask, position_ids=position_ids)[0]

            if i in self.attn_indices:
                # Apply Cross Attention
                cross1 = self.attn1_layers[f"layer_{i}"]
                cross2 = self.attn2_layers[f"layer_{i}"]
                
                # Update hidden states based on each other
                h1_new = cross1(query=h1, key=h2, value=h2, mask=processed_mask)
                h2_new = cross2(query=h2, key=h1, value=h1, mask=processed_mask)
                h1, h2 = h1_new, h2_new

        # 3. Output / CRF Logic
        emissions1 = self.emission1(h1)
        emissions2 = self.emission2(h2)
        
        mask_bool = attention_mask.bool() if attention_mask is not None else None

        if labels1 is not None and labels2 is not None:
            loss1 = -self.crf1(emissions1, labels1, mask=mask_bool, reduction='mean')
            loss2 = -self.crf2(emissions2, labels2, mask=mask_bool, reduction='mean')
            return loss1 + loss2
        
        return self.crf1.decode(emissions1, mask=mask_bool), \
               self.crf2.decode(emissions2, mask=mask_bool)