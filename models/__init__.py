from .mae import MaskedAutoencoderViT
from .simCLR import SimCLR
from .instance_simCLR import instanceSimCLR
from .classifier import Classifier

MODULES = { 
    "mae": MaskedAutoencoderViT,
    "simclr": SimCLR,
    "instance_simclr": instanceSimCLR,
    "classifier": Classifier
}