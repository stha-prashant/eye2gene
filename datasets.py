import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageBaseDataset(Dataset):
    def __init__(
        self,
        config,
    ):
        self.img_mean = (0.5, 0.5, 0.5)
        self.img_std = (0.5, 0.5, 0.5)
        self.img_size = config.im_size
        if config.transform_type == "train":
            self.transform = transforms.Compose(
                    [   
                        transforms.CenterCrop(288),
                        transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.img_mean, std=self.img_std),
                    ]
                )
        elif config.transform_type == "val" or config.transform_type =="test":
            self.transform  =transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.img_mean, std=self.img_std),
                ]
            )
        self.cfg = config

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def read_from_dicom(self, img_path):
        raise NotImplementedError


class KaggleOCTDatasetSingle(ImageBaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.root_dir = config.data.image.root_dir
        self.labels = sorted(os.listdir(self.root_dir))
        self.label_map = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}
        data = []
        for i in self.labels:
            base_path = os.path.join(self.root_dir, i)
            imgs = os.listdir(base_path)
            for img in imgs:
                img_path = base_path + '/' + img
                data.append(
                    {
                        "IMAGE": img_path,
                        "LABEL": self.label_map[i]
                    }
                )
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_dict = self.data[idx]
        image = Image.open(input_dict["IMAGE"]).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'label': input_dict["LABEL"]}