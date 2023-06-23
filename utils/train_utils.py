import random
import numpy as np
import torch
from tqdm import tqdm
from models.modules.mae_utils import merge_patches, mask_select


def select_device(device, batch_size):
    device = str(device).strip().lower()
    gpus = [int(gpu) for gpu in device.split(',')]
    n = len(gpus)
    if n > 1:
        assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
    return gpus

def fix_seed(random_seed=2023):
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def image_float2int(image):
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

def log_images(images, n=5):
    images = [np.array(x.detach().cpu()) for x in images]
    rows = np.concatenate(images, axis=2)
    #images = [x.reshape(-1, *x.shape[2:]) for x in images]
    #rows = np.concatenate(images, axis=2)
    rows = rows * 0.5 + 0.5
    return image_float2int(np.concatenate(rows[:n], axis=0))
def calculate_metrics_fromloader(model, dataloader, model_type, epoch=0):
    model.eval()
    avg_image_mse = 0.
    avg_text_ce = 0.
    with torch.no_grad():
        for batch in tqdm(dataloader):
            loss, _ = model.training_one_step(batch, epoch)
            if model_type == "multimodal":
                avg_image_mse += loss["Image Loss"]
                avg_text_ce += loss["Text loss"]
            elif model_type == "MAE":
                avg_image_mse += loss["Image Loss"]

    return avg_image_mse/len(dataloader), avg_text_ce/len(dataloader)

def pred_fn(model, dataloader, patch_size, epoch=0):
    model.eval()
    batch = next(iter(dataloader))

    _, images = model.training_one_step(batch, epoch)
    image_op, image_mask = images

    image = batch["image"]
    image = image.to(model.device)
    predicted_image = merge_patches(image_op, patch_size)
    predicted_image_combined = merge_patches(
                mask_select(mask=image_mask, this=model.patchify(image), other=image_op), patch_size
            )

    return image.permute((0, 2, 3, 1)), predicted_image, predicted_image_combined