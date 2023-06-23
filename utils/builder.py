import models
from torch.utils.data import DataLoader
import torch
from datasets import KaggleOCTDatasetSingle

def build_models(config, cuda, gpus, vocab_size=None):
    if config.phase == "pretraining":
        if config.model_type == "MAE":
            model = models.MODULES[config.architecture](
                    configs=config.model, 
                    device="cuda:" + str(gpus[0]) if cuda else "cpu"
            )
        return model
    if config.phase == "finetuning":
        if config.model_type == "MAE":
            base_model = models.MODULES[config.model.base_model.lower()](
               config.model.pretrained_params,
               device="cuda:" + str(gpus[0]) if cuda else "cpu"
            )
        if config.model.pretrained_path is not None:
            ckpt = torch.load(config.model.pretrained_path, map_location="cuda:" + str(gpus[0]))
            base_model.load_state_dict(ckpt['model'])
            for p in base_model.parameters():
                p.requires_grad = False
        return base_model
    raise NotImplementedError    
        
def build_datasets(config):
    if config.data.dataset == 'kaggle_oct_single':
        full_train_dataset = KaggleOCTDatasetSingle(config.data.train.root_dir)
        split_ratio = config.train.split_ratio
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [int(split_ratio*len(full_train_dataset)), len(full_train_dataset) - int(split_ratio*len(full_train_dataset))])
        test_dataset = KaggleOCTDatasetSingle(config.data.test.root_dir)
        return train_dataset, val_dataset, test_dataset
    raise NotImplementedError


def build_data(config):
    train_dataset, val_dataset, test_dataset = build_datasets(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.data.batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.data.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size)
    return train_loader, val_loader, test_loader
    
def build_optimizer(cfg, model):
    # get params for optimization
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if cfg.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.optimizer.weight_decay
        )
    

def build_loss(cfg):
    if cfg.loss_fn.type == "BCE":
        if cfg.loss_fn.class_weights is not None:
            weight = torch.Tensor(cfg.loss_fn.class_weights)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        return loss_fn
    else:
        raise NotImplementedError(f"{cfg.loss_fn.type} not implemented yet")