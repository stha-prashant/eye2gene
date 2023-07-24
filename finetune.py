import sys
import wandb
from omegaconf import OmegaConf
import argparse, os, time, datetime
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.train_utils import select_device, fix_seed, log_images, calculate_metrics_fromloader, pred_fn
from utils.builder import build_data, build_models, build_optimizer



parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file including parameters for model, data and training")
parser.add_argument("--gpus", type=str, help="gpu ids separated by comma")
parser.add_argument("--resume", type=str, default="", const=True, nargs='?', help="path to checkpoint")
parser.add_argument("--seed", type=int, default=2023, help="seeding for controlling randomization")
parser.add_argument("--plot_tsne", action="store_true", help="plot tSNE of embeddings")

args = parser.parse_args()

configs = OmegaConf.load(args.config)
args.logdir = os.path.join("/mnt/Enterprise/prashant/logs", configs.exp_name)
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
    
    if not args.plot_tsne:
        optimizer = build_optimizer(configs, model)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, 
                        T_0=configs.lr_warmup_epochs,
                        T_mult=configs.lr_mult)
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

if args.plot_tsne:
    from utils.train_utils import plot_tsne, plot_tsne_image, plot_tsne_real_fake
    plot_tsne_real_fake(model, val_loader, configs, f"val set, {configs.model_type} features")
    print("first done")
    # plot_tsne_image(model, val_loader, configs, "val set, PCA on image")
    exit(0)


prev_time = time.time()
runs = wandb.init(project=configs.wandb.project, entity=configs.wandb.username, name=configs.exp_name, reinit=True)
best_loss = float("inf")

for epoch in range(configs.epoch, configs.n_epoch):
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        loss, _ = model.training_one_step(batch)
        loss_med = loss["Loss"][0]/configs.grad_accum_steps
        loss_med.backward()
        
        if i%configs.grad_accum_steps == 0:
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
            wandb.log({"Loss": loss["Loss"][0]})
        else:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %s"
                % (
                    epoch,
                    configs.n_epoch,
                    i,
                    len(train_loader),
                    loss["Loss"][0].item(),
                    time_left,
                )
            )
            wandb.log({"Loss": loss["Loss"][0], "positive similarity": loss["Loss"][1], "negative similarity": loss["Loss"][2], "epoch": epoch})
        
    scheduler.step()
    
    val_mse, val_ce = calculate_metrics_fromloader(model=model, dataloader=val_loader, model_type=configs.model_type, epoch=epoch)

    if configs.model_type == "MAE":
        total_val_metric = val_mse
        wandb.log({
            "Val_image_mse": val_mse
        })
    else:
        total_val_metric = val_mse
        wandb.log({
            "Val_contrastive_loss": val_mse
        })

    if total_val_metric < best_loss:
        best_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
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

    if epoch % configs.log_freq == 0 and configs.model_type == "MAE":
        images = log_images(
            pred_fn(model, val_loader, configs.model.patch_sz, epoch),
            2
        )
        wandb.log({
            "image_prediction": wandb.Image(images)
        })
