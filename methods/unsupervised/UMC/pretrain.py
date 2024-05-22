import torch
import torch.nn.functional as F
import os
import logging

from tqdm import trange, tqdm
from transformers import BertTokenizer
from losses import loss_map
from utils.functions import save_model, restore_model
from .utils import * #set_optimizer, view_generator, get_pseudo_dataloader

from backbones.base import freeze_bert_parameters

class PretrainUMCManager:
    
    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        if args.freeze_pretrain_bert_parameters:
            self.logger.info('Freeze all parameters but the last layer for efficiency')
            self.model = freeze_bert_parameters(self.model, args.multimodal_method)

        self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr_pre)

        self.train_outputs = data.train_outputs
        
        self.contrast_criterion = loss_map['SupConLoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)

        if args.pretrain:
            
            self.logger.info('Pre-training start...')
            self._train(args)
            self.logger.info('Pre-training finished...')
            
        else:
            self.model = restore_model(self.model, os.path.join(args.model_output_path, 'pretrain'), self.device)
            
        self.model.to(torch.device('cpu'))
        torch.cuda.empty_cache()
        
    def _train(self, args):
        
        pseudo_data, pseudo_dataloader = get_pseudo_dataloader(args, self.train_outputs, mode='pretrain')

        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss, nb_tr_steps = 0, 0

            for batch in tqdm(pseudo_dataloader, desc = "Iteration"):
                
                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                
                with torch.set_grad_enabled(True):

                    mlp_output_a = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats, mode='pretrain-mm')
                    mlp_output_b = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device), mode='pretrain-mm')
                    mlp_output_c = self.model(text_feats, video_feats, audio_feats, mode='pretrain-mm')

                    norm_mlp_output_a = F.normalize(mlp_output_a)
                    norm_mlp_output_b = F.normalize(mlp_output_b)
                    norm_mlp_output_c = F.normalize(mlp_output_c)

                    contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1), norm_mlp_output_c.unsqueeze(1)), dim = 1)
                    loss_contrast_mm = self.contrast_criterion(contrastive_logits, temperature = args.pretrain_temperature, device = self.device)
                    loss = loss_contrast_mm
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                    
            loss = tr_loss / nb_tr_steps
            eval_results = {
                'train_loss': loss,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

        if args.save_model:
            pretrained_model_dir = os.path.join(args.model_output_path, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            save_model(self.model, pretrained_model_dir)
