"""Adapted from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1376"""
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel

class GPT2(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = GPT2Model.from_pretrained("gpt2", config=self.config)
        self.encoder.config.pad_token_id = self.encoder.config.eos_token_id
        self.classifier = nn.Linear(self.config.n_embd, self.config.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, x, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.encoder(x,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        
        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)
        
        batch_size = x.shape[0]

        # Ensure the batch size is > 1 if there is no padding.
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if x is not None:
                sequence_lengths = (torch.eq(x, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                print(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return pooled_logits
