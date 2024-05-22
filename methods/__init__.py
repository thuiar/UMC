from .unsupervised.USNID.manager import UnsupUSNIDManager
from .unsupervised.MCN.manager import MCNManager
from .unsupervised.CC.manager import CCManager
from .unsupervised.SCCL.manager import SCCLManager
from .unsupervised.UMC.manager import UMCManager

method_map = {
    'usnid': UnsupUSNIDManager,
    'mcn': MCNManager,
    'cc': CCManager,
    'sccl': SCCLManager,
    'umc': UMCManager,
}
