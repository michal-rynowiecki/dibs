from transformers import AutoModel

from torchcrf import CRF

from torch import nn

class TagModule(nn.Module):
    def __init__(self, model_path, num_tags):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

        hidden_size = self.bert.config.hidden_size

        self.emission_layer = nn.Linear(hidden_size, num_tags)

        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        emissions = self.emission_layer(sequence_output)

        # Tags known (training)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss

        # Tags unknown (inference)
        tag_sequence = self.crf.decode(emissions, mask=attention_mask.bool())
        return tag_sequence