# This is a file where I am trying to implement cross-attention between
# two simple neural networks

import torch
from torch import nn
import torch.nn.functional as F
import math

from transformers.models.bert.modeling_bert import BertModel, BertAttention

from transformers import AutoConfig, AutoModel, AutoTokenizer

'''
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.scale = 1.0 / math.sqrt(embed_dim)

        # Comes from encoder_new
        self.query = nn.Linear(embed_dim, embed_dim)

        # Comes from encoder_old
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_old, x_new):
        Q = self.query(x_new)
        K = self.key(x_old)
        V = self.value(x_old)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attention_weights = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, V)
        
        out = self.out_proj(out)
        print('Before norm: ', out)
        out = self.layer_norm(x_new + out)

        return out
'''

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.scale = 1.0 / math.sqrt(embed_dim)

        self.query = nn.Linear(embed_dim, embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) * self.scale

        attention_weights = F.softmax(attention_scores, dim=-1)
        
        out = torch.matmul(attention_weights, V)

        out = self.out_proj(out)

        out = self.layer_norm(x + out)
        return out

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim #// num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, L, H, Hd).transpose(1, 2)
        K = K.view(B, L, H, Hd).transpose(1, 2)
        V = V.view(B, L, H, Hd).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Hd)

        attention = scores.softmax(dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(out)

        return out



class BertWithExtraAttention(BertModel):
    def __init__(self, config, insert_after=0):
        super().__init__(config)
        self.insert_after = insert_after
        self.extra_attention = SelfAttention(config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        hidden_states = embedding_output

        print('Inputs: ', input_ids.shape)

        # Run each layer manually
        for i, layer_module in enumerate(self.encoder.layer):
            print(f'Hidden states layer {i}: {hidden_states.shape}')
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

            if i == self.insert_after:
                hidden_states = self.extra_attention(hidden_states)
        
        return hidden_states


if __name__=="__main__":
    model_path = "prajjwal1/bert-tiny"

    '''
    config = AutoConfig.from_pretrained(model_path)

    model = BertWithExtraAttention.from_pretrained(
        model_path,
        config=config,
        insert_after=0,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    inputs = tokenizer("hello world", return_tensors="pt")

    
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'].bool(),
        token_type_ids=inputs['token_type_ids']
    )
    print('Outputs: ', outputs.shape)
    '''
    attention_heads = MultiHeadedSelfAttention(4, 2)
    
    input = torch.rand(1, 4, 4)

    output = attention_heads(input)
    print(output.shape)