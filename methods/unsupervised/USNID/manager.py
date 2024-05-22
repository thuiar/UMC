import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time 


from sklearn.cluster import KMeans
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, set_torch_seed
from transformers import BertTokenizer

from utils.metrics import clustering_score
from .pretrain import PretrainUnsupUSNIDManager

from data.utils import get_dataloader
from data.base import get_data
from .utils import batch_chunk, get_augment_dataloader, _set_optimizer, view_generator

class UnsupUSNIDManager:
    
    def __init__(self, args, data, model):

        pretrain_manager = PretrainUnsupUSNIDManager(args, data, model)
    
        set_torch_seed(args.seed)

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model

        mm_dataloader = get_dataloader(args, data.mm_data)
    
        self.train_dataloader, self.test_dataloader = mm_dataloader['train'], mm_dataloader['test']

        self.train_outputs = data.train_outputs
        
        self.criterion = loss_map['CrossEntropyLoss']
        self.contrast_criterion = loss_map['SupConLoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)
        self.centroids = None
        
        if args.pretrain:
            self.pretrained_model = pretrain_manager.model

            self.num_labels = args.num_labels
            self.optimizer, self.scheduler = _set_optimizer(args, self.model, args.lr)
            self.load_pretrained_model(self.pretrained_model)
            
        else:
            self.num_labels = args.num_labels

            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.model_output_path, 'pretrain'), self.device)   

            if args.train:

                self.optimizer, self.scheduler = _set_optimizer(args, self.model, args.lr)
                if args.freeze_train_bert_parameters:
                    self.logger.info('Freeze all parameters but the last layer for efficiency')
                    self.model.freeze_bert_parameters(args.multimodal_method)
                
                self.load_pretrained_model(self.pretrained_model)
            else:
                self.model = restore_model(self.model, args.model_output_path, self.device)   

    def clustering(self, args, init = 'k-means++'):
        
        outputs = self._get_outputs(args, mode = 'train')
        feats = outputs['feats']
        
        if init == 'k-means++':
            
            self.logger.info('Initializing centroids with K-means++...')
            start = time.time()
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = 'k-means++').fit(feats) 
            
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            end = time.time()
            self.logger.info('K-means++ used %s s', round(end - start, 2))   
            
        elif init == 'centers':
            
            start = time.time()
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = self.centroids).fit(feats)
            km_centroids, assign_labels = km.cluster_centers_, km.labels_ 
            end = time.time()
            self.logger.info('K-means used %s s', round(end - start, 2))

        self.centroids = km_centroids
        pseudo_labels = torch.tensor(assign_labels, dtype=torch.long)      
        
        return pseudo_labels
                      
    def _train(self, args): 

        last_preds = None
        self.model.to(self.device)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            
            init_mechanism = 'k-means++' if epoch == 0 else 'centers'
            
            pseudo_labels = self.clustering(args, init = init_mechanism)

            current_preds = pseudo_labels.numpy()
            delta_label = np.sum(current_preds != last_preds).astype(np.float32) / current_preds.shape[0] 
            last_preds = np.copy(current_preds)
            
            if epoch > 0:
                
                self.logger.info("***** Epoch: %s *****", str(epoch))
                self.logger.info('Training Loss: %f', np.round(tr_loss, 5))
                self.logger.info('Delta Label: %f', delta_label)
                
                if delta_label < args.tol:
                    self.logger.info('delta_label %s < %f', delta_label, args.tol)  
                    self.logger.info('Reached tolerance threshold. Stop training.')
                    break                   
            
            self.train_outputs['label_ids'] = pseudo_labels

            pseudo_train_dataloader = get_augment_dataloader(self.generator, args, self.train_outputs, pseudo_labels)

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(pseudo_train_dataloader, desc="Training(All)"):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
            
                with torch.set_grad_enabled(True):
                    
                    text_feats_a, text_feats_b = batch_chunk(text_feats, dim=2)
                    label_ids = torch.chunk(input=label_ids, chunks=2, dim=1)[0][:, 0]

                    aug_mlp_output_a, aug_logits_a = self.model(text_feats_a, video_feats, audio_feats)
                    loss_model_a = self.model.get_model_loss(args.multimodal_method)
                    aug_mlp_output_b, aug_logits_b = self.model(text_feats_b, video_feats, audio_feats)
                    loss_model_b = self.model.get_model_loss(args.multimodal_method)

                    loss_model = (loss_model_a + loss_model_b) / 2
                    loss_ce = 0.5 * (self.criterion(aug_logits_a, label_ids) + self.criterion(aug_logits_b, label_ids))

                    norm_logits = F.normalize(aug_mlp_output_a)
                    norm_aug_logits = F.normalize(aug_mlp_output_b)
                    
                    contrastive_feats = torch.cat((norm_logits.unsqueeze(1), norm_aug_logits.unsqueeze(1)), dim = 1)
                    loss_contrast = self.contrast_criterion(contrastive_feats, labels = label_ids, temperature = args.train_temperature, device = self.device)
                    
                    loss = loss_contrast + loss_ce + loss_model
                    
                    self.optimizer.zero_grad()
                    loss.backward()

                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_loss += loss.item()
                    nb_tr_examples += label_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()
                
            tr_loss = tr_loss / nb_tr_steps
                
        if args.save_model:
            save_model(self.model, args.model_output_path)
              
    def _test(self, args):
        
        self.model.to(self.device)
        outputs = self._get_outputs(args, mode = 'test')
        feats = outputs['feats']
        y_true = outputs['y_true']

        km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, \
                    init = self.centroids if self.centroids is not None else 'k-means++').fit(feats) 
       
        y_pred = km.labels_
        
        test_results = clustering_score(y_true, y_pred)
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def _get_outputs(self, args, mode):
        
        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Get Outputs"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                features, logits = self.model(text_feats, video_feats, audio_feats, feature_ext=True)
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
        
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        
        y_logits = total_logits.cpu().numpy()
        
        outputs = {
            'y_true': y_true,
            'y_pred': y_pred,
            'logits': y_logits,
            'feats': feats
        }
        return outputs

    def load_pretrained_model(self, pretrained_model):
        
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['method_model.classifier.weight', 'method_model.classifier.bias',
                            'method_model.mlp_head.weight', 'method_model.mlp_head.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)
