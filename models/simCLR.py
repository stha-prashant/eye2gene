import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
from .modules.mae_utils import merge_patches, mask_select
import wandb
        
class SimCLR(nn.Module):
    def __init__(self, configs, device="cuda:0", mae_model=ModuleNotFoundError):
        super(SimCLR, self).__init__()
        
        self.device = device
        base_model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.proj = nn.Linear(base_model.fc.in_features, configs.image.proj_dim)
        self.mae_model = mae_model
        self.pass_through_separately = configs.pass_through_separately
        self.temperature = configs.temperature
        
    def forward(self, imgs):
        _, augmented_imgs, image_mask = self.mae_model(imgs)
        predicted_image_combined = merge_patches(
                    mask_select(mask=image_mask, this=self.mae_model.patchify(imgs), other=augmented_imgs), 16
                )

        augmented_imgs = predicted_image_combined.permute(0, 3, 1, 2)
        
        # alternative way 1: separately use encoder and proj
        if self.pass_through_separately:
            real_img_embeddings = self.encoder(imgs)
            reconstructed_img_embeddings = self.encoder(augmented_imgs)
            if self.proj is not None:
                reconstructed_img_embeddings = self.proj(reconstructed_img_embeddings.view(reconstructed_img_embeddings.shape[0], -1))
                real_img_embeddings = self.proj(real_img_embeddings.view(real_img_embeddings.shape[0], -1))

        # alternative way 2: combine real and reconstructed and use encoder and proj once
        else:
            img_embeddings = self.encoder(torch.cat([imgs, augmented_imgs], dim=0))
            if self.proj is not None:
                img_embeddings = self.proj(img_embeddings.view(img_embeddings.shape[0], -1))
            reconstructed_img_embeddings, real_img_embeddings = torch.split(img_embeddings, imgs.shape[0], dim=0)

        return reconstructed_img_embeddings, real_img_embeddings

    def forward_representation(self, imgs):
        # imgs = batch["image"]
        imgs = imgs.to(self.device)
        _, augmented_imgs, image_mask = self.mae_model(imgs)
        predicted_image_combined = merge_patches(
                    mask_select(mask=image_mask, this=self.mae_model.patchify(imgs), other=augmented_imgs), 16
                )

        augmented_imgs = predicted_image_combined.permute(0, 3, 1, 2)
        if self.pass_through_separately:
            reconstructed_img_embeddings = self.encoder(augmented_imgs)
            real_img_embeddings = self.encoder(imgs)
            if self.proj is not None:
                reconstructed_img_embeddings = self.proj(reconstructed_img_embeddings.view(reconstructed_img_embeddings.shape[0], -1))
                real_img_embeddings = self.proj(real_img_embeddings.view(real_img_embeddings.shape[0], -1))
        else:
            img_embeddings = self.encoder(torch.cat([imgs, augmented_imgs], dim=0))
            if self.proj is not None:
                img_embeddings = self.proj(img_embeddings.view(img_embeddings.shape[0], -1))
            reconstructed_img_embeddings, real_img_embeddings = torch.split(img_embeddings, imgs.shape[0], dim=0)
            
        return reconstructed_img_embeddings, real_img_embeddings
        
    
    def forward_loss(self, reconstructed_img_embeddings, real_img_embeddings):
        combined_imgs_embeddings = torch.cat([reconstructed_img_embeddings, real_img_embeddings], dim=0)
        features = nn.functional.normalize(combined_imgs_embeddings, dim=1)
        
        # in our case the first views are all positive and second view are all negative
        labels = torch.cat([torch.tensor([i]*reconstructed_img_embeddings.shape[0]) for i in range(2)], dim=0)
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
    
    
        # multiply all positives together to reduce to a single number per row
        # print(positives.shape)
        # positives = positives.prod(dim=1).view(positives.shape[0], -1)
        # # print(positives.shape)
        # # exit()

        # logits = torch.cat([positives, negatives], dim=1)
       
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
       
        # logits = logits / self.temperature
        # return nn.functional.cross_entropy(logits, labels)

    
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
        images = images.to(self.device)
        reconstructed_img_embeddings, real_img_embeddings = self(images)
        loss = self.forward_loss(reconstructed_img_embeddings, real_img_embeddings)
        return {"Loss": loss}, None