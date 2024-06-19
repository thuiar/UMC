import torch
import logging
from torch import nn
import torch.nn.functional as F
from .FusionNets import multimodal_methods_map
from .MethodNets import methods_map

__all__ = ['ModelManager']

def freeze_bert_parameters(model, multimodal_method):
    
    if multimodal_method in ['text']:
        for name, param in model.method_model.backbone.text_subnet.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    elif multimodal_method in ['mag_bert']:
        for name, param in model.method_model.backbone.model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    elif multimodal_method in ['mcn', 'umc']:
        for name, param in model.method_model.backbone.text_embedding.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    
    return model

class MIA(nn.Module):

    def __init__(self, args):

        super(MIA, self).__init__()

        fusion_method = multimodal_methods_map[args.multimodal_method]
        method_method = methods_map[args.method]
        fusion_backbone = fusion_method(args)
        self.method_model = method_method(args, fusion_backbone)
        
    def forward(self, text_feats, video_data, audio_data, *args, **kwargs):

        mm_model = self.method_model(text_feats, video_data, audio_data, *args, **kwargs)

        return mm_model



class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device)
        return model