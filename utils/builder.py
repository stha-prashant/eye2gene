import models
from torch.utils.data import DataLoader, Subset
import torch
from datasets import KaggleOCTDatasetMultiple, KaggleOCTDatasetSingle, KaggleOCTDatasetSingleContrastive, kaggle_oct_multiple_collate_fn

def build_models(config, cuda, gpus, vocab_size=None):
    if config.architecture == "simclr":
        ckpt = config.model.augmented_pretrained_path
        ckpt = torch.load(ckpt, map_location="cuda:" + str(gpus[0]))
        mae_model = models.MODULES[config.model.augmented_generating_model.lower()](
            config.model.pretrained_params,
            device="cuda:" + str(gpus[0]) if cuda else "cpu"
        )
        mae_model.load_state_dict(ckpt['model'])
        for p in mae_model.parameters():
            p.requires_grad = False

        model = models.MODULES[config.architecture](
            configs=config.model, 
            device="cuda:" + str(gpus[0]) if cuda else "cpu",
            mae_model = mae_model
        )
        return model


    if config.phase == "pretraining":
        model = models.MODULES[config.architecture](
                configs=config.model, 
                device="cuda:" + str(gpus[0]) if cuda else "cpu"
        )
        return model
    if config.phase == "finetuning":
        base_model = models.MODULES[config.model.base_model.lower()](
            config.model.pretrained_params,
            device="cuda:" + str(gpus[0]) if cuda else "cpu"
        )
        if config.model.pretrained_path is not None:
            ckpt = torch.load(config.model.pretrained_path, map_location="cuda:" + str(gpus[0]))
            base_model.load_state_dict(ckpt['model'])
            for p in base_model.parameters():
                p.requires_grad = False
        
        if config.architecture == "classifier":
            if config.model.pretrained_params.remove_proj_head:
                base_encoder = base_model.encoder

            model = models.MODULES[config.architecture](
                configs=config.model, 
                device="cuda:" + str(gpus[0]) if cuda else "cpu",
                base_encoder = base_encoder
            )

            return model
        return base_model
    raise NotImplementedError    
        
def build_datasets(config):
    if config.data.dataset == 'kaggle_oct_single':
        full_train_dataset = KaggleOCTDatasetSingle(config.data.train)
        split_ratio = config.data.split_ratio
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [int(split_ratio*len(full_train_dataset)), len(full_train_dataset) - int(split_ratio*len(full_train_dataset))])
        test_dataset = KaggleOCTDatasetSingle(config.data.test)
        return train_dataset, val_dataset, test_dataset, None
    if config.data.dataset == 'kaggle_oct_single_contrastive':
        full_train_dataset = KaggleOCTDatasetSingleContrastive(config.data.train)
        split_ratio = config.data.split_ratio
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [int(split_ratio*len(full_train_dataset)), len(full_train_dataset) - int(split_ratio*len(full_train_dataset))])
        test_dataset = KaggleOCTDatasetSingleContrastive(config.data.test)
        return train_dataset, val_dataset, test_dataset, None
    if config.data.dataset == 'kaggle_oct_multiple':
        full_train_dataset = KaggleOCTDatasetMultiple(config.data.train)
        split_ratio = config.data.split_ratio
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [int(split_ratio*len(full_train_dataset)), len(full_train_dataset) - int(split_ratio*len(full_train_dataset))])
        test_dataset = KaggleOCTDatasetMultiple(config.data.test)
        return train_dataset, val_dataset, test_dataset, kaggle_oct_multiple_collate_fn
    raise NotImplementedError


def build_data(config):
    train_dataset, val_dataset, test_dataset, custom_collate_fn = build_datasets(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers, collate_fn=custom_collate_fn, shuffle=True)

    try:
        if config.data.train_subset:
            ten_percent = int(config.data.train_subset * len(train_dataset))
            train_subset = Subset(train_dataset, torch.randperm(len(train_dataset))[:ten_percent])
            # Create a DataLoader for the train subset
            train_loader = DataLoader(train_subset, batch_size=config.data.batch_size, num_workers=config.data.num_workers, collate_fn=custom_collate_fn, shuffle=True)
    except:
        pass

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers, collate_fn=custom_collate_fn, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers, collate_fn=custom_collate_fn, shuffle=True)
    return train_loader, val_loader, test_loader
    
def define_param_groups(model, weight_decay, optimizer_name):
   def exclude_from_wd_and_adaptation(name):
       if 'bn' in name:
           return True
       if optimizer_name == 'lars' and 'bias' in name:
           return True

   param_groups = [
       {
           'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
           'weight_decay': weight_decay,
           'layer_adaptation': True,
       },
       {
           'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
           'weight_decay': 0.,
           'layer_adaptation': False,
       },
   ]
   return param_groups

def build_optimizer(cfg, model):
    # get params for optimization
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if cfg.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.optimizer.name == "AdamW":
        if cfg.model_type == 'SimCLR' or cfg.model_type == "instance_SimCLR":
            params = define_param_groups(model, cfg.optimizer.weight_decay, cfg.optimizer.name)
        return torch.optim.AdamW(
            params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
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