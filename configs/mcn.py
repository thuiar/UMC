class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """

        if args.multimodal_method == 'mcn':
            hyper_parameters = {
                'pretrained_model': 'uncased_L-12_H-768_A-12',
                'num_train_epochs': 200,
                'text_feat_dim': None,
                'video_feat_dim': None,
                'audio_feat_dim': None,
                't_dim': 768,
                'a_dim': 768,
                'v_dim': 256,
                'lr': [1e-4],
                'warmup_steps': 100,
                'embd_dim': [256],
                'recon_size': [128],
                'recon': True,
                'clu_lamb': [1],
                'recon_w': [50],
                'num_workers': 8,
                'feature_extractor_method': ['mean'],
                'freeze_train_bert_parameters': [True],
                'nheads': 8,
                'attn_dropout': 0.1,
                'relu_dropout': 0.1,
                'embed_dropout': 0.1,
                'res_dropout': 0.1,
                'attn_mask': True,
                'feats_processing_type': 'padding',
                'padding_mode': 'zero',
                'padding_loc': 'end',
                'train_batch_size': 128,
                'eval_batch_size': 64,
                'test_batch_size': 64,
                'wait_patience': 10,
            }      
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError
            
        return hyper_parameters