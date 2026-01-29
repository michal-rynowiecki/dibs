class SharedEncoderModule(nn.Module):
    def __init__(self, model_name, classes1, classes2, dropout_rate=0.1, class1_weights=None, class2_weights=None):
        super().__init__()
        # 1. Single Shared Encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 2. Separate Heads
        self.classifier1 = nn.Linear(hidden_size, len(classes1))
        self.classifier2 = nn.Linear(hidden_size, len(classes2))

        # Weights
        self.class1_weights = class1_weights if class1_weights is not None else torch.tensor([1.0]*len(classes1))
        self.class2_weights = class2_weights if class2_weights is not None else torch.tensor([1.0]*len(classes2))
        
        self.loss_fct1 = nn.CrossEntropyLoss(weight=self.class1_weights)
        self.loss_fct2 = nn.CrossEntropyLoss(weight=self.class2_weights)

    def forward(self, input_ids, attention_mask=None, labels1=None, labels2=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Predict both from the same features
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        if labels1 is not None and labels2 is not None:
            loss1 = self.loss_fct1(logits1, labels1)
            loss2 = self.loss_fct2(logits2, labels2)
            # Sum the losses (Multi-Task Learning)
            return loss1 + loss2 
        else:
            return logits1, logits2