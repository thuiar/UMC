import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

def get_augment_dataloader(args, train_outputs):

    text_data, video_data, audio_data = train_outputs['text'], train_outputs['video'], train_outputs['audio']
    train_data = MMPseudoDataset(text_data, video_data, audio_data)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size, drop_last=True)

    return train_dataloader

def _set_optimizer(args, model):
    
    if args.multimodal_method in ['text']:

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)

        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler

class MMPseudoDataset(Dataset):
        
    def __init__(self, text_feats, video_data, audio_data):
        
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(np.array(video_data['feats']))
        self.audio_feats = torch.tensor(np.array(audio_data['feats']))
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = { 
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index],
        } 
        return sample
