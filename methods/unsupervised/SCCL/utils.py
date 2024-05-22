import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def _set_optimizer(args, model, train_dataloader):
    
    cluster_centers = get_kmeans_centers(train_dataloader, args, model)
    model.method_model.init_model(cluster_centers=cluster_centers, alpha=args.alpha) 
    optimizer = torch.optim.Adam([
        {'params':model.method_model.backbone.parameters()}, 
        {'params':model.method_model.contrast_head.parameters(), 'lr': args.lr * args.lr_scale},
        {'params':model.method_model.cluster_centers, 'lr': args.lr * args.lr_scale}
    ], lr = args.lr)
    return optimizer 

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()

def get_kmeans_centers(train_loader, args, model):

    for i, batch in enumerate(tqdm(train_loader)):
        
        text_feats = batch['text_feats'].to(args.device)
        video_feats = batch['video_feats'].to(args.device)
        audio_feats = batch['audio_feats'].to(args.device)

        corpus_embeddings = model(text_feats, video_feats, audio_feats)
        if i == 0:     
            all_embeddings = corpus_embeddings.cpu().detach().numpy()
        else:
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.cpu().detach().numpy()), axis=0)

    print('embedding shape', all_embeddings.shape)
    clustering_model = KMeans(n_clusters=args.num_labels, random_state=args.seed)
    clustering_model.fit(all_embeddings)

    print("Iterations:{},  centers:{}".format(clustering_model.n_iter_,   clustering_model.cluster_centers_.shape))
    
    return clustering_model.cluster_centers_

def get_augment_dataloader(args, train_outputs):

    text_data, video_data, audio_data = train_outputs['text'], train_outputs['video'], train_outputs['audio']
    train_data = MMPseudoDataset(text_data, video_data, audio_data)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size, drop_last=True)

    return train_dataloader

class MMPseudoDataset(Dataset):
        
    def __init__(self, text_feats, video_data, audio_data):
        
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(np.array(video_data['feats']))
        self.audio_feats = torch.tensor(np.array(audio_data['feats']))
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = { 
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index],
        } 
        return sample

class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
            
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}
