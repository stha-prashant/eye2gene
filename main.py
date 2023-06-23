import sys
import wandb
from omegaconf import OmegaConf
import argparse, os, time, datetime
import torch
from torch.optim import lr_scheduler

from utils.train_utils import select_device, fix_seed, log_images, calculate_metrics_fromloader, pred_fn
from utils.builder import build_data, build_models, build_optimizer



parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file including parameters for model, data and training")
parser.add_argument("--gpus", type=str, help="gpu ids separated by comma")
parser.add_argument("--resume", type=str, default="", const=True, nargs='?', help="path to checkpoint")
parser.add_argument("--seed", type=int, default=2023, help="seeding for controlling randomization")


args = parser.parse_args()

configs = OmegaConf.load(args.config)
args.logdir = os.path.join("/raid/binod/prashant/eye2gene/logs", configs.exp_name)
os.makedirs(args.logdir, exist_ok=True)
os.makedirs(os.path.join(args.logdir, "images"), exist_ok=True)
os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)

fix_seed(args.seed)
args.cuda = torch.cuda.is_available()
if args.cuda:
    gpus = select_device(args.gpus, configs.data.batch_size)

train_loader, val_loader, test_loader = build_data(configs)
model = build_models(configs, args.cuda, gpus)

if args.resume:
    if not os.path.exists(args.resume):
        raise ValueError("Cannot find {}".format(args.resume))
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    optimizer = build_optimizer(configs, model)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer)
    scheduler.load_state_dict(ckpt['scheduler'])
    configs.epoch = ckpt['epoch']
else:
    optimizer = build_optimizer(configs, model)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer=optimizer,
                        T_0=configs.lr_warmup_epochs,
                        T_mult=configs.lr_mult,
                )

if args.cuda:
    model = model.float().cuda(gpus[0])

prev_time = time.time()
runs = wandb.init(project=configs.wandb.project, entity=configs.wandb.username, name=configs.exp_name, reinit=True)
best_loss = float("inf")

for epoch in range(configs.epoch, configs.n_epoch):
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss, _ = model.training_one_step(batch, epoch)
        loss["Loss"].backward()
        optimizer.step()

        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i
        batches_left = configs.n_epoch * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        # Print log
        if configs.model_type == "MAE":
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [image loss: %f] ETA: %s"
                % (
                    epoch,
                    configs.n_epoch,
                    i,
                    len(train_loader),
                    loss["Image Loss"].item(),
                    time_left,
                )
            )
            wandb.log(loss)
        
    scheduler.step()
    
    val_mse, val_ce = calculate_metrics_fromloader(model=model, dataloader=val_loader, model_type=configs.model_type, epoch=epoch)

    if configs.model_type == "MAE":
        total_val_metric = val_mse
        wandb.log({
            "Val_image_mse": val_mse
        })

    if total_val_metric < best_loss:
        best_ckpt = {
            "model": model.state_dict(),
            "epoch": epoch
        }
        torch.save(best_ckpt, os.path.join(args.logdir, "models", "best.pth"))
        best_loss = total_val_metric

    if epoch % configs.checkpoint_freq == 0:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(ckpt, os.path.join(args.logdir, "models", "latest.pth"))

    if epoch % configs.log_freq == 0:
        images = log_images(
            pred_fn(model, val_loader, configs.model.image.patch_size, epoch),
            2
        )
        wandb.log({
            "image_prediction": wandb.Image(images)
        })
