import torch
import logging
import time
from sklearn.cluster import KMeans
from tqdm import trange, tqdm
from utils.functions import restore_model, save_model
from utils.metrics import clustering_score
from data.utils import get_dataloader
from .utils import MMS_loss, _set_optimizer, get_pseudo_dataloader
from backbones.base import freeze_bert_parameters
__all__ = ['MCN']

class MCNManager:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model
        self.optimizer, self.scheduler = _set_optimizer(args, self.model)

        if args.freeze_train_bert_parameters:
            self.logger.info('Freeze all parameters but the last layer for efficiency')
            self.model = freeze_bert_parameters(self.model, args.multimodal_method)

        self.model.to(self.device)
        
        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['test']
        self.train_outputs = data.train_outputs
        
        self.num_labels = args.num_labels
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def clustering(self, args):
        
        feats, y_true = self._get_outputs(args, mode = 'train')
    
        start = time.time()
        self.logger.info('start kmeans...')
        if self.centroids is None:
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = 'k-means++').fit(feats) 
        else:
            km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, init = self.centroids.cpu().numpy()).fit(feats)
        end = time.time() 
        self.logger.info('K-means used %s s', round(end-start, 2))

        km_centroids, assign_labels = km.cluster_centers_, km.labels_
        self.centroids = torch.tensor(km_centroids).to(self.device)

        return assign_labels

    def _train(self, args): 

        self.centroids = None
        self.mms_loss = MMS_loss().to(self.device)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            tot_loss, tot_loss_mms, tot_loss_clu, tot_loss_recon, cnt = 0, 0, 0, 0, 0
            assigned_labels = self.clustering(args)
            self.model.train()
            pseudo_train_dataloader = get_pseudo_dataloader(args, self.train_outputs, assigned_labels)
            for batch in tqdm(pseudo_train_dataloader, desc="Training"):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device) 

                if args.recon:
                    text, video, audio, loss_recon = self.model(text_feats, video_feats, audio_feats, mode='train')
                else:
                    text, video, audio = self.model(text_feats, video_feats, audio_feats, mode='train')

                with torch.set_grad_enabled(True):
                    loss_mms = self._get_loss_mms(text, video, audio)
                    loss_clu = self._get_loss_cluster(text, video, audio, label_ids)
                    loss = loss_mms + loss_clu * args.clu_lamb
                    if args.recon:
                        loss += loss_recon * args.recon_w
                        tot_loss_recon += loss_recon.item()
                    self.optimizer.zero_grad()
                    loss.backward()

                    tot_loss += loss.item()
                    tot_loss_mms += loss_mms.item()
                    tot_loss_clu += loss_clu.item()
                    cnt += 1
                    
                 
                    self.optimizer.step()
                    self.scheduler.step()

            eval_results = {
                'tot_loss': round(tot_loss / cnt, 4),
                'tot_loss_mms': round(tot_loss_mms / cnt, 4),
                'tot_loss_clu': round(tot_loss_clu / cnt, 4),
            }
            if args.recon:
                eval_results['tot_loss_recon'] = round(tot_loss_recon / cnt, 4)


            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            if (epoch+1) % 50 == 0:
                self._test(args)

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)  
   
    def _get_outputs(self, args, mode):
        
        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, args.embd_dim)).to(self.device)

        for batch in tqdm(dataloader, desc="Getting outputs"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                text, video, audio = self.model(text_feats, video_feats, audio_feats, mode='eval')
                h = (text + video + audio) / 3
                total_features = torch.cat((total_features, h))
                total_labels = torch.cat((total_labels, label_ids))
                
        feats = total_features.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        return feats, y_true 
            
    def _test(self, args):
        
        self.model.to(self.device)
        feats, y_true = self._get_outputs(args, mode = 'test')

        km = KMeans(n_clusters = self.num_labels, n_jobs = -1, random_state=args.seed, \
                    init = self.centroids.cpu().numpy() if self.centroids is not None else 'k-means++').fit(feats) 
       
        y_pred = km.labels_
        
        test_results = clustering_score(y_true, y_pred)
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def cluster_contrast(self, fushed, labels):
        S = torch.matmul(fushed, self.centroids.t())
        target = torch.zeros(labels.shape[0], self.centroids.shape[0]).to(S.device)
        target[range(target.shape[0]), labels.type(torch.long)] = 1
        S = S - target * (0.001)
        S = S.view(S.shape[0], S.shape[1], -1)
        nominator = S * target[:, :, None]
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = S.view(S.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        I2C_loss = torch.mean(denominator - nominator)
        return I2C_loss

    def _get_loss_mms(self, text, video, audio):
        sim_audio_video = torch.matmul(audio, video.t())
        sim_audio_text = torch.matmul(audio, text.t())
        sim_text_video = torch.matmul(text, video.t())
        loss_mms = self.mms_loss(sim_text_video) + self.mms_loss(sim_audio_text) + self.mms_loss(sim_audio_video)
        return loss_mms

    def _get_loss_cluster(self, text, video, audio, labels):
        loss_clu = self.cluster_contrast(video, labels) + \
            self.cluster_contrast(audio, labels) + self.cluster_contrast(text, labels)
        loss_clu /= 3
        return loss_clu
