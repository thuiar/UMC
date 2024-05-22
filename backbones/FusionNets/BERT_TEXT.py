from torch import nn
from ..SubNets import text_backbones_map

class BERT_TEXT(nn.Module):

    def __init__(self, args):
        
        super(BERT_TEXT, self).__init__()
        text_backbone = text_backbones_map[args.text_backbone]
        if args.text_backbone == 'distilbert-base-nli-stsb-mean-tokens':
            self.text_subnet = text_backbone(args.pretrain_bert_model)[0].auto_model
        else:
            self.text_subnet = text_backbone(args)

    def forward(self, text_feats, video_feats, audio_feats):
        
        last_hidden_states = self.text_subnet(text_feats)
        features = last_hidden_states.mean(dim = 1)
        
        return features
