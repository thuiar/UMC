import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torch import optim
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

def _set_optimizer(args, model):

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs)
    return optimizer, scheduler


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MMS_loss(nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss

class MMPseudoDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_data, audio_data):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(np.array(video_data['feats']))
        self.audio_feats = torch.tensor(np.array(audio_data['feats']))
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index], 
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index],
        } 
        return sample

def get_pseudo_dataloader(args, train_outputs, pseudo_labels):
    train_text, train_video, train_audio = train_outputs['text'], train_outputs['video'], train_outputs['audio']
    pseudo_mm_train_data = MMPseudoDataset(pseudo_labels, train_text, train_video, train_audio)
    pseudo_train_dataloader = DataLoader(pseudo_mm_train_data, shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
    return pseudo_train_dataloader
