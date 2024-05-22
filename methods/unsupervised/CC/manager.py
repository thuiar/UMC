import torch
import logging
from utils.metrics import clustering_score
from backbones.base import freeze_bert_parameters
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from data.utils import get_dataloader
from utils.functions import save_model, restore_model
from .utils import _set_optimizer, get_augment_dataloader
from losses import contrastive_loss

class CCManager:
    
    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model
        self.num_labels = args.num_labels

        mm_dataloader = get_dataloader(args, data.mm_data)
   
        self.test_dataloader = mm_dataloader['test']
        self.augdataloader = get_augment_dataloader(args, data.train_outputs)
        self.instance_temperature = 0.7 
        self.cluster_temperature = 1.0
        self.criterion_instance = contrastive_loss.InstanceLoss(args.train_batch_size, self.instance_temperature, self.device) 
        self.criterion_cluster = contrastive_loss.ClusterLoss(self.num_labels, self.cluster_temperature, self.device) 

        if args.train:
            self.optimizer, self.scheduler = _set_optimizer(args, self.model)
            if args.freeze_train_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                self.model = freeze_bert_parameters(self.model, args.multimodal_method)
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)   

    def _get_outputs(self, args, mode='test'):
        
        if mode == 'test':
            dataloader = self.test_dataloader
            
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        
        for batch in tqdm(dataloader, desc='Get Outputs'):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
         
            with torch.set_grad_enabled(False):

                pooled_output = self.model(text_feats, video_feats, audio_feats)
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))
 
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        outputs = {
            'feats': feats,
            'y_true': y_true,
        }
        return outputs

    def _train(self, args):
         
        self.logger.info('CC training starts...')
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            tr_loss, nb_tr_steps = 0, 0
            self.model.train()
            for batch in tqdm(self.augdataloader, desc="Training(All)"):
                
                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                                               
                with torch.set_grad_enabled(True):
                    
                    x_i = self.model(text_feats, video_feats, audio_feats)
                    loss_model_a = self.model.get_model_loss(args.multimodal_method)
                    x_j = self.model(text_feats, video_feats, audio_feats)
                    loss_model_b = self.model.get_model_loss(args.multimodal_method)
                    loss_model = (loss_model_a + loss_model_b) / 2

                    z_i, z_j, c_i, c_j = self.model.method_model.get_features(x_i, x_j)
                    loss_instance = self.criterion_instance(z_i, z_j)
                    loss_cluster = self.criterion_cluster(c_i, c_j)
                    loss = loss_instance + loss_cluster + loss_model

                    self.optimizer.zero_grad()
                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_steps += 1
                                
                    self.optimizer.step()
                    self.scheduler.step()
                
            train_loss = tr_loss / nb_tr_steps
            
            self.logger.info("***** Epoch: %s: train results *****", str(epoch))
            self.logger.info("  train_loss = %s",  str(train_loss))
        
        self.logger.info('CC training finished...')
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
