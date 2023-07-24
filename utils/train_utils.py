import random
import numpy as np
import torch
from tqdm import tqdm
from models.modules.mae_utils import merge_patches, mask_select
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os

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
            else:
                avg_image_mse += loss["Loss"][0]

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


def plot_tsne(model, data_loader, config, title=''): 
    model.eval()
    X = []
    X_labels = []
    for batch in tqdm(data_loader):
        features = model.forward_representation(batch["image"].to(model.device))
        # features = features[:, 1, :] # get cls features only
        # if config.model_type == "MAE":
        #     features = torch.mean(features, dim=1) # get avg pooled features
        # elif config.model_type == "multimodal":
        #     features = features
        # else:
        #     features = features[1]


        # perform pca to reduce to 50 dimensions
        

        labels = batch["label"].to(model.device)
        X.append(features.detach().cpu().numpy())
        X_labels.append(labels.detach().cpu().numpy())

    # get X and X labels
    X = np.concatenate(X, axis=0)
    X_labels = np.concatenate(X_labels, axis=0)
    print(X.shape, X_labels.shape)
    
    k = 4

    # Create a KMeans object
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # Get the labels for each data point
    kmeans_labels = kmeans.labels_
    
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=40)
    # X = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(X)
    dftsne = pd.DataFrame(tsne_embeddings)
    dftsne['cluster'] = pd.Series(X_labels)
    dftsne.columns = ['x1','x2','cluster']
    os.makedirs(f"output/{config.model_type}", exist_ok=True)
    dftsne_kmeans = dftsne.copy()
    dftsne_kmeans['cluster'] = kmeans_labels
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",palette="deep", alpha=0.5).set_title(title + ", cluster colors using ground truth")
    plt.savefig(f"output/{config.model_type}/tsne_gt.png")
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=dftsne_kmeans,x='x1', y='x2', hue='cluster', legend='full', palette="deep", alpha=0.5).set_title(title+", cluster colors using k means")
    plt.savefig(f"output/{config.model_type}/tsne_kmeans.png")

    print(title, metrics.rand_score(dftsne['cluster'], dftsne_kmeans['cluster']))
    print(title, metrics.adjusted_rand_score(dftsne['cluster'], dftsne_kmeans['cluster']))

def plot_tsne_real_fake(model, data_loader, config, title=''): 
    model.eval()
    X = []
    X_labels = []
    for i, batch in tqdm(enumerate(data_loader)):
        features = model.forward_representation(batch["image"].to(model.device))
        # features = features[:, 1, :] # get cls features only
        if config.model_type == "MAE":
            features = torch.mean(features, dim=1) # get avg pooled features
        else:
            fake = features[0]
            real = features[1]
            
        features = torch.cat((fake, real), dim=0)
        labels = torch.cat((torch.zeros(fake.shape[0]), torch.ones(real.shape[0])), dim=0)
        X.append(features.detach().cpu().numpy())
        X_labels.append(labels.detach().cpu().numpy())
    
    # get X and X labels
    X = np.concatenate(X, axis=0)
    X_labels = np.concatenate(X_labels, axis=0)
    print(X.shape, X_labels.shape)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(X)
    dftsne = pd.DataFrame(tsne_embeddings)
    dftsne['cluster'] = pd.Series(X_labels)
    dftsne.columns = ['x1','x2','cluster']

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",palette="deep", alpha=0.5).set_title(title + ", cluster colors using ground truth")
    plt.savefig(f"output/{config.model_type}/tsne_gt_realvfake.png")






def plot_tsne_image(model, data_loader, config, title=''): 
    model.eval()
    X = []
    X_labels = []
    for batch in tqdm(data_loader):
        # features = model.forward_representation(batch["image"].to(model.device))
        # features = features[:, 1, :] # get cls features only
        # features = torch.mean(features, dim=1) # get avg pooled features
        features = batch["image"].to(model.device)
        features = features.reshape(features.shape[0], -1)
        

        labels = batch["label"].to(model.device)
        X.append(features.detach().cpu().numpy())
        X_labels.append(labels.detach().cpu().numpy())

    # get X and X labels
    X = np.concatenate(X, axis=0)
    X_labels = np.concatenate(X_labels, axis=0)
    print(X.shape, X_labels.shape)
    
    k = 4

    # Create a KMeans object
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # Get the labels for each data point
    kmeans_labels = kmeans.labels_
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=384)
    X = pca.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne.fit_transform(X)
    dftsne = pd.DataFrame(tsne_embeddings)
    dftsne['cluster'] = pd.Series(X_labels)
    dftsne.columns = ['x1','x2','cluster']
    
    dftsne_kmeans = dftsne.copy()
    dftsne_kmeans['cluster'] = kmeans_labels
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=dftsne,x='x1',y='x2',hue='cluster',legend="full",palette="deep", alpha=0.5).set_title(title + ", cluster colors using ground truth")
    plt.savefig(f"output/{config.model_type}/tsne_gt_image.png")
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=dftsne_kmeans,x='x1', y='x2', hue='cluster', legend='full', palette="deep", alpha=0.5).set_title(title+", cluster colors using k means")
    plt.savefig(f"output/{config.model_type}/tsne_kmeans_image.png")
    
    print(title, metrics.rand_score(dftsne['cluster'], dftsne_kmeans['cluster']))
    print(title, metrics.adjusted_rand_score(dftsne['cluster'], dftsne_kmeans['cluster']))