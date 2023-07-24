import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

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
                        # transforms.CenterCrop(224),
                        # transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=self.img_mean, std=self.img_std),
                    ]
                )
        elif config.transform_type == "val" or config.transform_type =="test":
            self.transform  =transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=self.img_mean, std=self.img_std),
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
        self.root_dir = config.root_dir
        self.labels = sorted(os.listdir(self.root_dir))
        self.label_map = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}
        data = []
        for i in self.labels:
            base_path = os.path.join(self.root_dir, i)
            imgs = os.listdir(base_path)
            for img in imgs:
                # if image is not png or jpg, skip
                if img.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
                    continue
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

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class KaggleOCTDatasetSingleContrastive(ImageBaseDataset):
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def __init__(self, config):
        super().__init__(config)
        self.root_dir = config.root_dir
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
        self.transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32),2)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_dict = self.data[idx]
        # print(input_dict)
        # exit()
        image = Image.open(input_dict["IMAGE"]).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'label': input_dict["LABEL"]}


def get_files_with_parent_folder(path):
    files = os.listdir(path)
    files_with_parent_folder = [os.path.join(path, file)  for file in files]
    return files_with_parent_folder

class KaggleOCTDatasetMultiple(ImageBaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.root_dir = config.root_dir
        self.classes = sorted(os.listdir(self.root_dir))
        self.per_patient_images = config.per_patient_images
        self.valid_patient_ids = self._filter_invalid_data()
        self.label_map = {'CNV':0, 'DME':1, 'DRUSEN':2, 'NORMAL':3}

    def _filter_invalid_data(self):
        patient_ids = self._get_patient_ids()
        # iterate over patient_ids
        valid_patient_ids = []
        for i, patient_id in enumerate(patient_ids):
            patient_folder = patient_id
            image_files = os.listdir(patient_folder)
            if len(image_files) < self.per_patient_images:
                continue
            valid_patient_ids.append(patient_id)
        return valid_patient_ids


    def _get_patient_ids(self):
        patient_ids = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            patient_ids.extend(get_files_with_parent_folder(class_dir))
        return patient_ids

    def __len__(self):
        return len(self.valid_patient_ids)

    def __getitem__(self, idx):
        patient_id = self.valid_patient_ids[idx]
        class_name = patient_id.split("/")[-2]
        patient_folder = patient_id
        image_files = os.listdir(patient_folder)

        images = []

        for i in range(self.per_patient_images):
            img_name = image_files[i]
            img_path = os.path.join(patient_folder, img_name)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        return {'image': images, 'label': self.label_map[class_name]}
    
def kaggle_oct_multiple_collate_fn(batch):
    assert len(batch) == 1
    images = []
    labels = []
    for i in batch:
        images.extend(i['image'])
        labels.append(i['label'])
    return {'image': torch.stack(images), 'label': torch.tensor(labels).repeat(len(images), 1).squeeze(1)}