import logging
import numpy as np
import copy
import torch
import torch.nn as nn

from utils.metrics import clustering_score
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from utils.functions import save_model
from data.utils import get_dataloader
from .utils import _set_optimizer, target_distribution, get_augment_dataloader, PairConLoss

class SCCLManager:
    
    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        self.num_labels = args.num_labels
        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['test']
        self.train_outputs = data.train_outputs

        self.augdataloader = get_augment_dataloader(args, self.train_outputs)
        # 
        self.model.to(self.device)

        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=args.temperature)

        if args.train:
            # self.optimizer, self.scheduler = _set_optimizer(args, self.model)
            self.optimizer = _set_optimizer(args, self.model, self.train_dataloader)
            if args.freeze_train_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                self.model = freeze_bert_parameters(self.model, args.multimodal_method)
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)  

    def _train(self, args):

        self.logger.info('SCCL training starts...')
 
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  
            self.model.train()
            
            tr_loss, nb_tr_steps = 0, 0
            for batch in tqdm(self.augdataloader, desc="Training(All)"):
                with torch.set_grad_enabled(True):

                    text_feats = batch['text_feats'].to(self.device)
                    video_feats = batch['video_feats'].to(self.device)
                    audio_feats = batch['audio_feats'].to(self.device)

                    embd1 = self.model(text_feats, video_feats, audio_feats)
                    loss_model_1 = self.model.get_model_loss(args.multimodal_method)
                    embd2 = self.model(text_feats, video_feats, audio_feats)
                    loss_model_2 = self.model.get_model_loss(args.multimodal_method)
                    embd3 = self.model(text_feats, video_feats, audio_feats)
                    loss_model_3 = self.model.get_model_loss(args.multimodal_method)
                    loss_model = (loss_model_1 + loss_model_2 + loss_model_3) / 3 
                    # Instance-CL loss
                    feat1, feat2 = self.model.method_model.contrast_logits(embd2, embd3)
                    losses = self.contrast_loss(feat1, feat2)
                    loss = args.eta * losses["loss"] + loss_model

                    output = self.model.method_model.get_cluster_prob(embd1)
                    target = target_distribution(output).detach()
                    cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
                    loss += cluster_loss
                    losses["cluster_loss"] = cluster_loss.item()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    tr_loss += loss.item()
                    nb_tr_steps += 1

            train_loss = tr_loss / nb_tr_steps
            self.logger.info("***** Epoch: %s: train results *****", str(epoch))
            self.logger.info("  train_loss = %s",  str(train_loss))

        self.logger.info('SCCL training finished...')

        if args.save_model:
            save_model(self.model, args.model_output_path)


    def _test(self, args):
        
        outputs = self._get_outputs(args, mode = 'test')
        feats, y_true = outputs['feats'], outputs['y_true']

        km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed).fit(feats)
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
            
        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                features = self.model(text_feats, video_feats, audio_feats)
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
     
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = {
            'feats': feats,
            'y_true': y_true,
        }
        return outputs
