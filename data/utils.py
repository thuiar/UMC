import pickle
import numpy as np
import os

from torch.utils.data import DataLoader

def get_dataloader(args, data):

    train_dataloader = DataLoader(data['train'], shuffle=False, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
    test_dataloader = DataLoader(data['test'], batch_size = args.test_batch_size, num_workers = args.num_workers, pin_memory = True)

    return {
        'train': train_dataloader,
        'test': test_dataloader
    }  

def get_v_a_data(data_args, feats_path, max_seq_len):
    
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of features is empty.')    

    feats = load_feats(data_args, feats_path)
    data = padding_feats(feats, max_seq_len)
    
    return data 
    
def load_feats(data_args, video_feats_path):

    with open(video_feats_path, 'rb') as f:
        video_feats = pickle.load(f)

    train_feats = [video_feats[x] for x in data_args['train_data_index']]
    test_feats = [video_feats[x] for x in data_args['test_data_index']]
    outputs = {
        'train': train_feats,
        'test': test_feats
    }

    return outputs

def padding(feat, max_length, padding_mode = 'zero', padding_loc = 'end'):
    """
    padding_mode: 'zero' or 'normal'
    padding_loc: 'start' or 'end'
    """
    assert padding_mode in ['zero', 'normal']
    assert padding_loc in ['start', 'end']

    length = feat.shape[0]
    if length > max_length:
        return feat[:max_length, :]

    if padding_mode == 'zero':
        pad = np.zeros([max_length - length, feat.shape[-1]])
    elif padding_mode == 'normal':
        mean, std = feat.mean(), feat.std()
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))
    
    if padding_loc == 'start':
        feat = np.concatenate((pad, feat), axis = 0)
    else:
        feat = np.concatenate((feat, pad), axis = 0)

    return feat

def padding_feats(feats, max_seq_len):

    p_feats = {}

    for dataset_type in feats.keys():
        f = feats[dataset_type]

        tmp_list = []
        length_list = []
        
        for x in f:
            x_f = np.array(x) 
            x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f

            length_list.append(len(x_f))
            p_feat = padding(x_f, max_seq_len)
            tmp_list.append(p_feat)

        p_feats[dataset_type] = {
            'feats': tmp_list,
            'lengths': length_list
        }

    return p_feats    
