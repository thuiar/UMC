import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from data.mm_pre import MMDataset
import numpy as np
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

def get_pseudo_dataloader(args, train_outputs, mode='pretrain', pseudo_labels=None):
    
    '''
    video_data: {'feats': xxx, 'lengths': xxx}
    audio_data: {'feats': xxx, 'lengths': xxx}
    text_data: [input_ids, input_mask, segment_ids]
    '''
    text = train_outputs['text']
    video = train_outputs['video']
    audio = train_outputs['audio'] 

    if 'select_ids' in train_outputs.keys():
        select_ids = train_outputs['select_ids']    
                                           
    if pseudo_labels is None:
        pseudo_labels = train_outputs['label_ids']

    if 'select_ids' in train_outputs.keys():

        pseudo_labels = pseudo_labels[select_ids]
        text = [text[i] for i in select_ids]
        new_video = {}

        new_video['lengths'] = [video['lengths'][i] for i in select_ids]
        new_video['feats'] = [video['feats'][i] for i in select_ids]

        new_audio = {}
        new_audio['lengths'] = [audio['lengths'][i] for i in select_ids]
        new_audio['feats'] = [audio['feats'][i] for i in select_ids]

        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
        train_data = MMDataset(train_label_ids, text, new_video, new_audio)

    else:
        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
        train_data = MMDataset(train_label_ids, text, video, audio)

    sampler = RandomSampler(train_data)

    if mode == 'pretrain':
        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.pretrain_batch_size)
    else:
        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)
                                      
    return train_data, train_dataloader

class view_generator:
    
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
    
    def random_token_erase(self, input_x, input_mask, max_seq_length=30, mode = 'text'):
        
        aug_input_x = []
        aug_input_mask = []

        for inp_x, inp_m in zip(input_x, input_mask):
            
            if mode == 'text':
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp_x, already_has_special_tokens=True)
                sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]

                inds = np.arange(len(sent_tokens_inds))
                masked_inds = np.random.choice(inds, size = int(len(inds) * self.args.re_prob), replace = False)
                sent_masked_inds = sent_tokens_inds[masked_inds]

                inp_x = np.delete(inp_x, sent_masked_inds)
                inp_x = F.pad(inp_x, (0, max_seq_length - len(inp_x)), mode = 'constant', value = 0)

                inp_m = np.delete(inp_m, sent_masked_inds)
                inp_m = F.pad(inp_m, (0, max_seq_length - len(inp_m)), 'constant', 0)
            
            else:
                sent_tokens_inds = np.where(inp_m.numpy() == 1)[0]

                erase_start_ind = np.random.choice(sent_tokens_inds)
                erase_end_ind = min(erase_start_ind + max(1, int(self.args.re_prob * (len(sent_tokens_inds) - erase_start_ind))), len(sent_tokens_inds))
                erase_inds = sent_tokens_inds[erase_start_ind: erase_end_ind]

                inp_x = np.delete(inp_x, erase_inds, axis = 0)
                inp_x = torch.from_numpy(inp_x)
                inp_x = F.pad(inp_x, (0, 0, 0, max_seq_length - len(inp_x)), mode = 'constant', value = 0)

                inp_m = np.delete(inp_m, erase_inds)
                inp_m = F.pad(inp_m, (0, max_seq_length - len(inp_m)), 'constant', 0)

            aug_input_x.append(inp_x)
            aug_input_mask.append(inp_m)
        
        aug_input_x = torch.stack(aug_input_x, dim = 0)
        aug_input_mask = torch.stack(aug_input_mask, dim = 0)

        return aug_input_x, aug_input_mask

        
def set_optimizer(args, model, lr, mode='train'):
    
    num_train_epochs = args.num_train_epochs 

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = lr, correct_bias=False)

    num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * num_train_epochs
    num_warmup_steps = int(args.num_train_examples * num_train_epochs * args.warmup_proportion / args.train_batch_size)
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler