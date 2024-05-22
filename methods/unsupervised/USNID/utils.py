import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from data.mm_pre import MMDataset
import numpy as np
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

def batch_chunk(x, dim=1):
    x1, x2 = torch.chunk(input=x, chunks=2, dim=dim)
    x1, x2 = x1.squeeze(dim), x2.squeeze(dim)
    return x1, x2

def get_augment_dataloader(generator, args, train_outputs, pseudo_labels = None):

    text_data = train_outputs['text']
    video_data = train_outputs['video']
    audio_data = train_outputs['audio']

    text_data = torch.tensor(text_data)
    input_ids, input_mask, segment_ids = text_data[:, 0], text_data[:, 1], text_data[:, 2]

    if pseudo_labels is None:
        pseudo_labels = train_outputs['label_ids']
    
    input_ids_a, input_mask_a = generator.random_token_erase(input_ids, input_mask)
    input_ids_b, input_mask_b = generator.random_token_erase(input_ids, input_mask)

    train_input_ids = torch.cat(([input_ids_a.unsqueeze(1), input_ids_b.unsqueeze(1)]), dim = 1).tolist()
    train_input_mask = torch.cat(([input_mask_a.unsqueeze(1), input_mask_b.unsqueeze(1)]), dim = 1).tolist()
    train_segment_ids = torch.cat(([segment_ids.unsqueeze(1), segment_ids.unsqueeze(1)]), dim = 1).tolist()

    train_text_feats = [[train_input_ids[i], train_input_mask[i], train_segment_ids[i]] for i in range(len(train_input_ids))]

    train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
    train_label_ids = torch.cat(([train_label_ids, train_label_ids]), dim = 1)
    
    train_data = MMDataset(train_label_ids, train_text_feats, video_data, audio_data)

    sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)

    return train_dataloader

def _set_optimizer(args, model, lr):
    
    if args.multimodal_method in ['text', 'mag_bert', 'cmc']:

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr = lr, correct_bias=False)

        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
    
    elif args.multimodal_method in ['misa']:

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    return optimizer, scheduler

class view_generator:
    
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
    
    def random_token_erase(self, input_ids, input_mask, audio_feats=None, video_feats=None):

        aug_input_ids = []
        aug_input_mask = []
        
        for inp_i, inp_m in zip(input_ids, input_mask):
            
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp_i, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            masked_inds = np.random.choice(inds, size = int(len(inds) * self.args.re_prob), replace = False)
            sent_masked_inds = sent_tokens_inds[masked_inds]

            inp_i = np.delete(inp_i, sent_masked_inds)
            inp_i = F.pad(inp_i, (0, self.args.text_seq_len - len(inp_i)), 'constant', 0)

            inp_m = np.delete(inp_m, sent_masked_inds)
            inp_m = F.pad(inp_m, (0, self.args.text_seq_len - len(inp_m)), 'constant', 0)
    
            aug_input_ids.append(inp_i)
            aug_input_mask.append(inp_m)
        
        aug_input_ids = torch.stack(aug_input_ids, dim=0)
        aug_input_mask = torch.stack(aug_input_mask, dim=0)

        return aug_input_ids, aug_input_mask