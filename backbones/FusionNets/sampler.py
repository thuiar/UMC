import torch
import numpy as np
from torch import nn

class ConvexSampler(nn.Module):
    
    def __init__(self, args):
        super(ConvexSampler, self).__init__()
        self.ood_label_id = args.ood_label_id
        self.args = args

    def forward(self, text, video, audio, mm, label_ids, device = None):
        
        num_ood = len(text) * self.args.multiple_ood
        
        ood_text_list, ood_video_list, ood_audio_list, ood_mm_list = [], [], [], []
        
        text_seq_length, video_seq_length, audio_seq_length, mm_seq_length = \
            text.shape[1], video.shape[1], audio.shape[1], mm.shape[1]
        
        if label_ids.size(0) > 2:
            
            while len(ood_text_list) < num_ood:
                cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                
                if label_ids[cdt[0]] != label_ids[cdt[1]]:
                    
                    s = np.random.uniform(0, 1, 1)
                    
                    ood_text_list.append(s[0] * text[cdt[0]] + (1 - s[0]) * text[cdt[1]])
                    ood_video_list.append(s[0] * video[cdt[0]] + (1 - s[0]) * video[cdt[1]])
                    ood_audio_list.append(s[0] * audio[cdt[0]] + (1 - s[0]) * audio[cdt[1]])
                    ood_mm_list.append(s[0] * mm[cdt[0]] + (1 - s[0]) * mm[cdt[1]])

            ood_text = torch.cat(ood_text_list, dim = 0).view(num_ood, text_seq_length, -1)
            ood_video = torch.cat(ood_video_list, dim = 0).view(num_ood, video_seq_length, -1)
            ood_audio = torch.cat(ood_audio_list, dim = 0).view(num_ood, audio_seq_length, -1)
            ood_mm = torch.cat(ood_mm_list, dim = 0).view(num_ood, mm_seq_length)
            
            mix_text = torch.cat((text, ood_text), dim = 0)
            mix_video = torch.cat((video, ood_video), dim = 0)
            mix_audio = torch.cat((audio, ood_audio), dim = 0)
            mix_mm = torch.cat((mm, ood_mm), dim = 0)
            # print('00000000000', mix_text.shape)
            semi_label_ids = torch.cat((label_ids.cpu(), torch.tensor([self.ood_label_id] * num_ood)), dim=0)
            # print('11111111', semi_label_ids.shape)
            
        mix_data = {}
        mix_data['text'] = mix_text.to(device)
        mix_data['video'] = mix_video.to(device)
        mix_data['audio'] = mix_audio.to(device)
        mix_data['mm'] = mix_mm.to(device)
        
        mix_labels = {
            'ind': label_ids.to(device),
            'semi': semi_label_ids.to(device)
        }
       
        return mix_data, mix_labels

