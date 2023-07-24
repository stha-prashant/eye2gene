import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
from .modules.mae_utils import merge_patches, mask_select
import wandb

class instanceSimCLR(nn.Module):
    def __init__(self, configs, device="cuda:0"):
        super(instanceSimCLR, self).__init__()
        
        self.device = device
        base_model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.proj = nn.Linear(base_model.fc.in_features, configs.image.proj_dim)
        self.temperature = configs.temperature
        
    def forward(self, imgs):
        real_img_embeddings = self.encoder(imgs)
        real_img_embeddings = self.proj(real_img_embeddings.view(real_img_embeddings.shape[0], -1))
        return real_img_embeddings

    def forward_representation(self, imgs):
        # imgs = batch["image"]
        imgs = imgs.to(self.device)
        real_img_embeddings = self.encoder(imgs).view(imgs.shape[0], -1)
        real_img_embeddings = self.proj(real_img_embeddings)
       
        return real_img_embeddings
        
    
    def forward_loss(self, img_embeddings):
        features = nn.functional.normalize(img_embeddings, dim=1)
        
        # in our case the first views are all positive and second view are all negative
        labels = torch.cat([torch.arange(img_embeddings.shape[0]/2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        similarity_matrix = torch.matmul(features, features.T)
       
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

       
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)


        nominator = torch.exp(positives / self.temperature)
        denominator = torch.exp(negatives / self.temperature).sum(dim=1, keepdim=True)
        all_losses = -torch.log(nominator / denominator)
        # print(nominator[0])
        # print(denominator[0])
        all_losses = all_losses.mean()
        return all_losses, positives.mean(), negatives.mean()
    
    def evaluate(self, dataloader):
        self.eval()
        avg_loss = 0.
        with torch.no_grad():
            for batch in tqdm(dataloader):
                loss, _ = self.training_one_step(batch)
                avg_loss += loss["Loss"]

        return {
            "dev total": avg_loss / len(dataloader),
        }
    
    
    def training_one_step(self, batch, epoch=None):
        images = batch["image"]
        images = torch.cat(images, dim=0)
        images = images.to(self.device)
        img_embeddings = self(images)
        loss = self.forward_loss(img_embeddings)
        return {"Loss": loss}, None