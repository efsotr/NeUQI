import torch
from transformers.data import default_data_collator

from datautils import get_loaders
from utils.arguments_init import get_data_args, get_tokenizer


def data_collator(*args, **kwargs):
    ret = default_data_collator(*args, **kwargs)
    if ret["input_ids"].shape[0] == 1:
        max_length = ret["input_ids"].shape[1]
        ret["max_length_q"] = max_length
        ret["max_length_k"] = max_length
        ret["cu_seq_lens_q"] = torch.tensor([0, max_length], dtype=torch.int, device=ret["input_ids"].device)
        ret["cu_seq_lens_k"] = ret["cu_seq_lens_q"]
    return ret

def get_data():
    data_args = get_data_args()
    tokenizer = get_tokenizer()

    train_dataloader = get_loaders(name=data_args.dataset,
                                   nsamples=data_args.nsamples,
                                   seqlen=data_args.seqlen,
                                   tokenizer_path=tokenizer)
    train_dataloader = [{"input_ids": train_dataloader[i][0]} for i in range(len(train_dataloader))]

    test_datasets = sorted(set(data_args.test_dataset.split(",")))
    test_dataloaders = {}
    for test_dataset in test_datasets:
        test_dataloader =  get_loaders(name=test_dataset,
                                      seqlen=data_args.seqlen,
                                      tokenizer_path=tokenizer,
                                      eval_mode=True)
        nsamples = test_dataloader.numel() // data_args.seqlen
        test_dataloader = test_dataloader.view(-1)[: nsamples * data_args.seqlen].view(nsamples, data_args.seqlen)
        test_dataloader = [{"input_ids": test_dataloader[i]} for i in range(nsamples)]
        test_dataloaders[test_dataset] = test_dataloader

    return train_dataloader, test_dataloaders, data_collator
