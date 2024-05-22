from .MCN import MCN
from .UMC import UMC
from .BERT_TEXT import BERT_TEXT
multimodal_methods_map = {
    'mcn': MCN,
    'umc': UMC,
    'text': BERT_TEXT,
}