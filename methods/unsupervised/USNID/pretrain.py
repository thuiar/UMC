import torch
import torch.nn.functional as F
import os
import logging

from tqdm import trange, tqdm
from transformers import BertTokenizer
from losses import loss_map
from utils.functions import save_model, restore_model
from .utils import batch_chunk, get_augment_dataloader, _set_optimizer, view_generator


class PretrainUnsupUSNIDManager:
    
    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        if args.freeze_pretrain_bert_parameters:
            self.logger.info('Freeze all parameters but the last layer for efficiency')
            self.model.freeze_bert_parameters(args.multimodal_method)

        self.optimizer, self.scheduler = _set_optimizer(args, self.model, args.lr_pre)

        self.train_outputs = data.train_outputs
        
        self.contrast_criterion = loss_map['SupConLoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)

        if args.pretrain:
            
            self.logger.info('Pre-raining start...')
            self._train(args)
            self.logger.info('Pre-training finished...')
            
        else:
            self.model = restore_model(self.model, os.path.join(args.model_output_path, 'pretrain'), self.device)
            
        self.model.to(torch.device('cpu'))
        torch.cuda.empty_cache()
        
    def _train(self, args):
        
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            
            self.model.train()
            tr_loss, nb_tr_steps = 0, 0

            contrast_dataloader = get_augment_dataloader(self.generator, args, self.train_outputs)

            for batch in tqdm(contrast_dataloader, desc = "Iteration"):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                
                with torch.set_grad_enabled(True):

                    text_feats_a, text_feats_b = batch_chunk(text_feats, dim=2)
                    aug_mlp_output_a, _ = self.model(text_feats_a, video_feats, audio_feats)
                    loss_model_a = self.model.get_model_loss(args.multimodal_method)
                    aug_mlp_output_b, _ = self.model(text_feats_b, video_feats, audio_feats)   
                    loss_model_b = self.model.get_model_loss(args.multimodal_method)           

                    norm_logits = F.normalize(aug_mlp_output_a)
                    norm_aug_logits = F.normalize(aug_mlp_output_b)

                    contrastive_logits = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    
                    loss_contrast = self.contrast_criterion(contrastive_logits, temperature = args.pretrain_temperature, device = self.device)
                    loss = loss_contrast + (loss_model_a + loss_model_b) / 2
                    
                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.zero_grad()
                    loss.backward()
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
