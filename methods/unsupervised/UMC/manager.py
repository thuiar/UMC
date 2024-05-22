import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time 
import copy

from sklearn.cluster import KMeans
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, set_torch_seed
from transformers import BertTokenizer

from backbones.base import freeze_bert_parameters
from sklearn.neighbors import NearestNeighbors, KDTree
from utils.neighbor_dataset import NeighborsDataset
from torch.utils.data import DataLoader

from utils.metrics import clustering_score
from .pretrain import PretrainUMCManager

from data.utils import get_dataloader
from .utils import *

class UMCManager:
    
    def __init__(self, args, data, model):

        pretrain_manager = PretrainUMCManager(args, data, model)

        set_torch_seed(args.seed)

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model

        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.test_dataloader = mm_dataloader['train'], mm_dataloader['test']
        self.train_outputs = data.train_outputs
        
        self.criterion = loss_map['CrossEntropyLoss']
        self.contrast_criterion = loss_map['SupConLoss']
        self.mse_criterion = loss_map['MSELoss']

        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)    
        self.generator = view_generator(self.tokenizer, args)
        self.centroids = None

        if args.pretrain:
            self.pretrained_model = pretrain_manager.model

            self.num_labels = args.num_labels
            self.load_pretrained_model(self.pretrained_model)
            
        else:
            self.num_labels = args.num_labels
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.model_output_path, 'pretrain'), self.device)   
            
        if args.train:
            args.num_train_epochs = (1 - args.thres) / args.delta
            self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr)

            if args.freeze_train_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                self.model = freeze_bert_parameters(self.model, args.multimodal_method)
            
            self.load_pretrained_model(self.pretrained_model)
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)   

    def clustering(self, args, init = 'k-means++', threshold = 0.25):
        
        outputs = self._get_outputs(args, mode = 'train', return_feats = True)
        feats = outputs['feats']
        y_true = outputs['y_true']
        
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

        select_ids = []

        for cluster_id in range(self.num_labels):
            cluster_samples = feats[assign_labels == cluster_id]
            pos = list(np.where(assign_labels == cluster_id)[0])
            
            cutoff = max(int(len(cluster_samples) * threshold), 1)
            k_candidate_proportions = np.arange(0.1, 0.32, 0.02).tolist()

            if cutoff == 1:
                select_ids.extend(pos)

            else:
                
                best_sorted_indices = None
                best_eval_score = 0
                best_k_cand = None

                for k_cand in k_candidate_proportions:

                    k = max(int(len(cluster_samples) * k_cand), 1)
                    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(cluster_samples)
                    distances, indices = nbrs.kneighbors(cluster_samples)
                    
                    reachable_distances = np.mean(distances[:, 1:], axis=1) 
                    density = 1 / reachable_distances  
                    sorted_indices = np.argsort(density)

                    tmp_select_indices = sorted_indices[-cutoff:]
                    tmp_select_pos = [pos[i] for i in tmp_select_indices] 
                    
                    tmp_feats = feats[tmp_select_pos]
                    tmp_assign_labels = assign_labels[tmp_select_pos]

                    tree = KDTree(tmp_feats)
                    _, ind = tree.query(tmp_feats, k=2)  
                    nearest_neighbor_distances = np.array([tmp_feats[i] - tmp_feats[ind[i, 1]] for i in range(len(tmp_feats))])
                    tmp_eval_score = np.mean(np.linalg.norm(nearest_neighbor_distances, axis=1))

                    if tmp_eval_score > best_eval_score:
               
                        best_eval_score = tmp_eval_score
                        best_k_cand = k_cand
                        best_sorted_indices = sorted_indices

                select_indices = best_sorted_indices[-cutoff:]
                select_pos = [pos[i] for i in select_indices] 
                select_ids.extend(select_pos)
        

        return np.array(assign_labels), select_ids, feats
                  
    def _train(self, args): 
        
        self.model.to(self.device)
        
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):

            threshold = args.thres + args.delta * epoch

            init_mechanism = 'k-means++' if epoch == 0 else 'centers'
            pseudo_labels, select_ids, feats = self.clustering(args, init = init_mechanism, threshold = threshold)

            if epoch > 0:
                self.logger.info("***** Epoch: %s *****", str(epoch))
                self.logger.info('Supervised Training Loss: %f', np.round(tr_sup_loss, 5))
                if len(non_select_ids) != 0:
                    self.logger.info('Unsupervised Training Loss: %f', np.round(tr_unsup_loss, 5))

            self.train_outputs['label_ids'] = pseudo_labels
            self.train_outputs['select_ids'] = select_ids
            
            _, pseudo_sup_train_dataloader = get_pseudo_dataloader(args=args, \
                train_outputs=self.train_outputs, mode='pretrain')
            
            
            self.model.train()
            tr_sup_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            
            for batch_sup in tqdm(pseudo_sup_train_dataloader, desc = 'Iteration'):

                text_feats = batch_sup['text_feats'].to(self.device)
                video_feats = batch_sup['video_feats'].to(self.device)
                audio_feats = batch_sup['audio_feats'].to(self.device)

                label_ids = batch_sup['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    _, mlp_output_a = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats, mode='train-mm')
                    _, mlp_output_b = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device), mode='train-mm')
                    _, mlp_output_c = self.model(text_feats, video_feats, audio_feats, mode='train-mm')

                    norm_mlp_output_a = F.normalize(mlp_output_a)
                    norm_mlp_output_b = F.normalize(mlp_output_b)
                    norm_mlp_output_c = F.normalize(mlp_output_c)

                    contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1), norm_mlp_output_c.unsqueeze(1)), dim = 1)
                    loss_sup = self.contrast_criterion(contrastive_logits, labels = label_ids, temperature = args.train_temperature_sup, device = self.device)

                    loss = loss_sup

                    self.optimizer.zero_grad()
                    loss.backward()


                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    tr_sup_loss += loss.item()
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.scheduler.step()

                    torch.cuda.empty_cache()


            tr_sup_loss /= nb_tr_steps

            non_select_ids = [i for i in range(len(pseudo_labels)) if i not in select_ids]
            unsup_pseudo_labels = pseudo_labels[non_select_ids]

            if len(non_select_ids) != 0:

                self.train_outputs['select_ids'] = non_select_ids
                _, pseudo_unsup_train_dataloader = get_pseudo_dataloader(args=args, \
                    train_outputs=self.train_outputs, mode='pretrain')

                tr_unsup_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                
                for batch_unsup in tqdm(pseudo_unsup_train_dataloader, desc = 'Iteration'):

                    unsup_text_feats = batch_unsup['text_feats'].to(self.device)
                    unsup_video_feats = batch_unsup['video_feats'].to(self.device)
                    unsup_audio_feats = batch_unsup['audio_feats'].to(self.device)

                    with torch.set_grad_enabled(True):
    
                        _, mlp_output_a = self.model(unsup_text_feats, torch.zeros_like(unsup_video_feats).to(self.device), unsup_audio_feats, mode='train-mm')
                        _, mlp_output_b = self.model(unsup_text_feats, unsup_video_feats, torch.zeros_like(unsup_audio_feats).to(self.device), mode='train-mm')
                        _, mlp_output_c = self.model(unsup_text_feats, unsup_video_feats, unsup_audio_feats, mode='train-mm')

                        norm_mlp_output_a = F.normalize(mlp_output_a)
                        norm_mlp_output_b = F.normalize(mlp_output_b)
                        norm_mlp_output_c = F.normalize(mlp_output_c)

                        contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1), norm_mlp_output_c.unsqueeze(1)), dim = 1)
                        loss_unsup = self.contrast_criterion(contrastive_logits, temperature = args.train_temperature_unsup, device = self.device)
                    
                        loss = loss_unsup

                        self.optimizer.zero_grad()
                        loss.backward()


                        if args.grad_clip != -1.0:
                            torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                        tr_unsup_loss += loss.item()
                        nb_tr_steps += 1

                        self.optimizer.step()
                        self.scheduler.step()

                        torch.cuda.empty_cache()
                
                tr_unsup_loss /= nb_tr_steps

        if args.save_model:
            save_model(self.model, args.model_output_path)

    def _test(self, args):
        
        self.model.to(self.device)

        outputs = self._get_outputs(args, mode = 'test', return_feats = True)
        feats = outputs['feats']
        y_true = outputs['y_true']

        km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, \
                    init = 'k-means++').fit(feats) 
       
        y_pred = km.labels_
        
        test_results = clustering_score(y_true, y_pred)

        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def _get_outputs(self, args, mode, return_feats = False, modality = 'tva'):
        
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
                if modality == 'tva':
                    features, _ = self.model(text_feats, video_feats, audio_feats, mode='train-mm')
                elif modality == 't0a':
                    features, _ = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats, mode='train-mm')
                elif modality == 'tv0':
                    features, _ = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device), mode='train-mm')


                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
        
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        
        if return_feats:
            outputs = {
                'y_true': y_true,
                'feats': feats
            }

        else:
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
        mlp_params = ['method_model.mlp_head_train.2.weight', 'method_model.mlp_head_train.2.bias', 'method_model.classifier.weight', 'method_model.classifier.bias']
  
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in mlp_params}
        self.model.load_state_dict(pretrained_dict, strict=False)