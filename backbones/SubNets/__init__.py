from sentence_transformers import SentenceTransformer
from .FeatureNets import BERTEncoder, RoBERTaEncoder

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'roberta-base': RoBERTaEncoder,
                    'distilbert-base-nli-stsb-mean-tokens': SentenceTransformer,
                }
