"""Adapted from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/albert/modeling_albert.py#L1025"""
import torch.nn as nn
from transformers import AlbertModel, AlbertPreTrainedModel

class Albert(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = AlbertModel.from_pretrained("albert-base-v2", config=self.config)
        self.dropout = nn.Dropout(self.config.classifier_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.encoder(x,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
