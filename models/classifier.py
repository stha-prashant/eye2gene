import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt
from .modules.mae_utils import merge_patches, mask_select
import wandb

class Classifier(nn.Module):
    def __init__(self, configs, device="cuda:0", base_encoder=None):
        print(type(self), Classifier)
        super(Classifier, self).__init__()
        
        self.device = device
        self.encoder = base_encoder
        if base_encoder is None:
            base_model = models.resnet50(pretrained=True)
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        else:
            if configs.freeze_encoder:
                # freeze the encoder
                for param in self.encoder.parameters():
                    param.requires_grad = False
            else:
                # unfreeze the encoder
                for param in self.encoder.parameters():
                    param.requires_grad = True
            

        self.classifier_head = nn.Linear(configs.encoder_output_dim, configs.num_classes)
        
    def forward(self, imgs):
        real_img_embeddings = self.encoder(imgs).view(imgs.shape[0], -1)
        return self.classifier_head(real_img_embeddings)    
    
    def evaluate(self, dataloader):
        self.eval()
        avg_loss = 0.
        accuracy = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                loss, _ = self.training_one_step(batch)
                avg_loss += loss["Loss"]
                accuracy += loss["batch_accuracy"]

        

        return {
            "dev total": avg_loss / len(dataloader),
            "dev accuracy": accuracy / len(dataloader),
        }
    
    
    def training_one_step(self, batch, epoch=None):
        images = batch["image"]
        labels = batch["label"]
        images = images.to(self.device)
        output = self(images)
        loss = nn.CrossEntropyLoss()(output, labels.to(self.device))
        batch_accuracy = (output.argmax(dim=1) == labels.to(self.device)).float().mean()
        return {"Loss": loss, "batch_accuracy": batch_accuracy}, None