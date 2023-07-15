from .mae import MaskedAutoencoderViT
from .simCLR import SimCLR

MODULES = { 
    "mae": MaskedAutoencoderViT,
    "simclr": SimCLR,
}