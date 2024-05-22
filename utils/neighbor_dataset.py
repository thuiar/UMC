import torch
import numpy as np
from torch.utils.data import Dataset

class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()

        self.dataset = dataset
        self.indices = indices 
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]

        output['anchor'] = {}
        output['anchor']['text'] = anchor['text_feats']
        output['anchor']['video'] = anchor['video_feats']
        output['anchor']['audio'] = anchor['audio_feats']
        output['anchor']['video_lengths'] = anchor['video_lengths']
        output['anchor']['audio_lengths'] = anchor['audio_lengths']
        output['anchor']['label_ids'] = anchor['label_ids']
        
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])

        output['index'] = index

        return output