from .USNID import USNIDModel
from .MCN import MCNModel
from .CC import CCModel
from .SCCL import SCCLModel
from .UMC import UMCModel

methods_map = {
    'usnid': USNIDModel,
    'mcn': MCNModel,
    'cc': CCModel,
    'sccl': SCCLModel,
    'umc': UMCModel,
}
