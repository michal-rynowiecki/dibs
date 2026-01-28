class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
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

        return self.norm(query + self.dropout(attn_output))


class DualModule(nn.Module):
    def __init__(self, model1, model2, num_tags, attn_layers=None):
        super().__init__()

        self.bert1 = AutoModel.from_pretrained(model1)
        self.bert2 = AutoModel.from_pretrained(model2)

        hidden_size = self.bert1.config.hidden_size
        num_layers = self.bert1.config.num_hidden_layers

        self.attn_layers = attn_layers if attn_layers is not None else [num_layers - 1]

        self.attn1_layers = nn.ModuleDict()
        self.attn2_layers = nn.ModuleDict()

        for layer_idx in self.attn_layers:
            self.attn1_layers[str(layer_idx)] = CrossAttentionLayer(hidden_size)
            self.attn2_layers[str(layer_idx)] = CrossAttentionLayer(hidden_size)

        self.emission1 = nn.Linear(hidden_size, num_tags)
        self.emission2 = nn.Linear(hidden_size, num_tags)

        self.crf1 = CRF(num_tags, batch_first=True)
        self.crf2 = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels1=None, labels2=None):

        # Run full forward pass (DeBERTa-safe)
        out1 = self.bert1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        out2 = self.bert2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states1 = out1.hidden_states
        hidden_states2 = out2.hidden_states

        # Default: last layer
        hidden1 = hidden_states1[-1]
        hidden2 = hidden_states2[-1]

        # Apply cross-attention on selected layers
        for layer_idx in self.attn_layers:
            h1 = hidden_states1[layer_idx]
            h2 = hidden_states2[layer_idx]

            h1 = self.attn1_layers[str(layer_idx)](
                query=h1, key=h2, value=h2, mask=attention_mask
            )
            h2 = self.attn2_layers[str(layer_idx)](
                query=h2, key=h1, value=h1, mask=attention_mask
            )

            hidden1 = h1
            hidden2 = h2

        emissions1 = self.emission1(hidden1)
        emissions2 = self.emission2(hidden2)

        if labels1 is not None and labels2 is not None:
            loss1 = -self.crf1(
                emissions1, labels1,
                mask=attention_mask.bool(),
                reduction='mean'
            )
            loss2 = -self.crf2(
                emissions2, labels2,
                mask=attention_mask.bool(),
                reduction='mean'
            )
            return loss1 + loss2

        tags1 = self.crf1.decode(emissions1, mask=attention_mask.bool())
        tags2 = self.crf2.decode(emissions2, mask=attention_mask.bool())

        return tags1, tags2
