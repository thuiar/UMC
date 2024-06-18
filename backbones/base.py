import torch
import logging
from torch import nn
import torch.nn.functional as F
from .FusionNets import multimodal_methods_map
from .MethodNets import methods_map

__all__ = ['ModelManager']

def freeze_bert_parameters(model, multimodal_method):
    
    if multimodal_method in ['text', 'misa']:
        for name, param in model.method_model.backbone.text_subnet.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    elif multimodal_method in ['mag_bert']:
        for name, param in model.method_model.backbone.model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    elif multimodal_method in ['mcn', 'cmc', 'umc']:
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



    def get_model_loss(self, multimodal_method):

        if multimodal_method in ['misa']:

            diff_loss = self.method_model.backbone._get_diff_loss()
            domain_loss = self.method_model.backbone._get_domain_loss()
            recon_loss = self.method_model.backbone._get_recon_loss()
            cmd_loss = self.method_model.backbone._get_cmd_loss()
            
            if self.method_model.backbone.args.use_cmd_sim:
                similarity_loss = cmd_loss
            else:
                similarity_loss = domain_loss

            loss = self.method_model.backbone.args.diff_weight * diff_loss + \
                self.method_model.backbone.args.sim_weight * similarity_loss + \
                self.method_model.backbone.args.recon_weight * recon_loss
            return loss
        
        elif multimodal_method in ['cmc']:

            backbone = self.method_model.backbone
            if backbone.loss_mode == 'rdrop':
                l1 = backbone._compute_kl_loss(backbone.logits1, backbone.logits2)
                l2 = backbone._compute_kl_loss(backbone.logits1, backbone.logits3)
                l3 = backbone._compute_kl_loss(backbone.logits2, backbone.logits3)
                loss_rdrop = l1 + l2 + l3
                tot_loss = backbone.weight * loss_rdrop

            elif backbone.loss_mode == 'mse':
                mse = F.mse_loss(backbone.logits1, backbone.logits) + \
                    F.mse_loss(backbone.logits2, backbone.logits) + \
                    F.mse_loss(backbone.logits3, backbone.logits)
                tot_loss = backbone.weight * mse

            else:
                tot_loss = 0.0
            
            return tot_loss

        elif multimodal_method in ['mag_bert', 'text', 'umc']:
            
            return 0.0
        
class ModelManager:

    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args) 
        model.to(self.device)
        return model