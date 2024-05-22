import os
import logging
import csv
import copy

from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        bm = benchmarks[args.dataset]
        max_seq_lengths, feat_dims = bm['max_seq_lengths'], bm['feat_dims']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = max_seq_lengths['text'], max_seq_lengths['video'], max_seq_lengths['audio']
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims['text'], feat_dims['video'], feat_dims['audio']
        
        self.mm_data, self.train_outputs = get_data(args, self.logger) 
        
def get_data(args, logger):
    
    data_path = os.path.join(args.data_path, args.dataset)
    bm = benchmarks[args.dataset]
    
    label_list = copy.deepcopy(bm["labels"])
    logger.info('Lists of intent labels are: %s', str(label_list))  
      
    args.num_labels = len(label_list)  
    
    logger.info('data preparation...')
    
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'train.tsv'))
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'dev.tsv'))
    
    train_data_index = train_data_index + dev_data_index
    train_label_ids = train_label_ids + dev_label_ids

    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'test.tsv'))
    args.num_train_examples = len(train_data_index)
    
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'test_data_index': test_data_index,
    }
        
    text_data = get_t_data(args, data_args)
        
    video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
    video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)
    
    audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
    audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)
    
    mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
    mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

    mm_data = {'train': mm_train_data, 'test': mm_test_data}
    
    train_outputs = {
        'text': text_data['train'],
        'video': video_data['train'],
        'audio': audio_data['train'],
        'label_ids': train_label_ids,
    }
    
    return mm_data, train_outputs
                 
def get_indexes_annotations(args, bm, label_list, read_file_path):

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if args.dataset in ['MIntRec']:
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                
                label_id = label_map[line[4]]
            
            elif args.dataset in ['MELD-DA']:
                label_id = label_map[line[3]]
                
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
            
            elif args.dataset in ['IEMOCAP-DA']:
                label_id = label_map[line[2]]
                index = line[0]
                indexes.append(index)
            
            label_ids.append(label_id)
    
    return indexes, label_ids