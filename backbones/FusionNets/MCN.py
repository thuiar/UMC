import torch
from torch import nn
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder

__all__ = ['MCN']

class MCN(nn.Module):
    
    def __init__(self, args):

        super(MCN, self).__init__()

        self.embd_dim = args.embd_dim
        self.recon = args.recon
        
        self.feature_extractor_method = args.feature_extractor_method
        if self.feature_extractor_method == 'cnn':
            self.audio_conv = nn.Conv1d(480, 1, kernel_size=1, padding=0)
            self.video_conv = nn.Conv1d(230, 1, kernel_size=1, padding=0)
        elif self.feature_extractor_method == 'trfm':
            self.num_heads = args.nheads
            self.attn_dropout = args.attn_dropout
            self.relu_dropout = args.relu_dropout
            self.embed_dropout = args.embed_dropout
            self.res_dropout = args.res_dropout
            self.attn_mask = args.attn_mask
            self.audio_trfm = self.get_transformer_encoder(args.a_dim, 3)
            self.video_trfm = self.get_transformer_encoder(args.v_dim, 3)
        
        if self.recon:
            self.recon_size = args.recon_size
            self.recon_v = nn.Sequential(
                nn.Linear(args.embd_dim, self.recon_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.recon_size, args.v_dim),
                nn.ReLU(inplace=True)
            )
            self.recon_a = nn.Sequential(
                nn.Linear(args.embd_dim, self.recon_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.recon_size, args.a_dim),
                nn.ReLU(inplace=True)
            )
            self.recon_t = nn.Sequential(
                nn.Linear(args.embd_dim, self.recon_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.recon_size, args.embd_dim),
                nn.ReLU(inplace=True)
            )
            self.mse = nn.MSELoss(reduction='none')
        
        self.GU_audio = Gated_Embedding_Unit(args.a_dim, args.embd_dim)
        self.GU_video = Gated_Embedding_Unit(args.v_dim, args.embd_dim)
        self.text_pooling_caption = Sentence_Maxpool(args.t_dim, args.embd_dim)
        self.GU_text_captions = Gated_Embedding_Unit(args.embd_dim, args.embd_dim)
        self.text_embedding = BERTEncoder(args)
    
    def get_transformer_encoder(self, embed_dim, layers):
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
        
    def forward(self, text, video, audio, mode='train'): 
        text = self.text_embedding(text)
        # what to do with audio? using mean? TODO
        video = torch.tensor(video, dtype=torch.float)
        audio = torch.tensor(audio, dtype=torch.float)
        if self.feature_extractor_method == 'mean':
            video = video.mean(dim=1)  # temporary use mean, actually view(-1,shape[-1]), choices: mean, Conv1d, LSTM, Transformer
            audio = audio.mean(dim=1)  # this averages features from 0 padding too  
        elif self.feature_extractor_method == 'cnn':
            video = self.video_conv(video).squeeze(1)
            audio = self.audio_conv(audio).squeeze(1)
        elif self.feature_extractor_method == 'trfm':
            video = video.transpose(0, 1)
            audio = audio.transpose(0, 1)
            video = self.video_trfm(video)[-1]
            audio = self.audio_trfm(audio)[-1]
        else:
            raise NotImplementedError        

        text_gt = self.text_pooling_caption(text)
        text = self.GU_text_captions(text_gt)
        
        if mode == 'train' and self.recon:
            video_gt = video
            video = self.GU_video(video)
            video_recon = self.recon_v(video)
            audio_gt = audio
            audio = self.GU_audio(audio)
            audio_recon = self.recon_a(audio)
            text_recon = self.recon_t(text)
            mse_v = torch.mean(self.mse(video_recon, video_gt), dim=-1)
            mse_a = torch.mean(self.mse(audio_recon, audio_gt), dim=-1)
            mse_t = torch.mean(self.mse(text_recon, text_gt), dim=-1)
            loss_recon = mse_v + mse_a + mse_t
            loss_recon = torch.mean(loss_recon)
            return text, video, audio, loss_recon
        else:
            video = self.GU_video(video)
            audio = self.GU_audio(audio)
            return text, video, audio

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()

        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
    
        x = self.fc(x)
        x = self.cg(x)
        return x

class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)

class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return torch.max(x, dim=1)[0]