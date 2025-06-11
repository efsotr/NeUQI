import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


@torch.compile
def cross_entropy(*args, **kwargs):
    return F.cross_entropy(*args, **kwargs)

def combine(model: PreTrainedModel, quantized_model: PreTrainedModel):

    def wrap(fn):
        def wrapped_fn(input_ids, labels=None, **kwargs):
            if labels is None:
                hidden_states = model(input_ids, need_logits=False, **kwargs).logits
                torch.cuda.empty_cache()
                qhidden_states = fn(input_ids, need_logits=False, **kwargs).logits
                return (hidden_states - qhidden_states).square().mean()

            logits = fn(input_ids=input_ids, **kwargs).logits
            labels = torch.nn.functional.pad(labels, pad=(0, 1), value=-100)[..., 1:].contiguous()
            losses = cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
            losses = losses.view(labels.shape).float().sum(dim=-1) / (labels.size(-1) - 1)
            return losses

        return wrapped_fn

    quantized_model.forward = wrap(quantized_model.forward)
