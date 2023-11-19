import os

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cv2
import numpy as np


        
class UltrasonicDataset(Dataset):
    def __init__(self, root, transforms):
        super().__init__()

        self.root = root

        self.images = []
        self.masks = []

        self.make_dataset()

        self.transforms = transforms

    
    def make_dataset(self):
        images_path = os.path.join(self.root, 'image')
        masks_path = os.path.join(self.root, 'mask')

        for image_name in sorted(os.listdir(images_path)):
            image_path = os.path.join(images_path, image_name)
            mask_path = os.path.join(masks_path, image_name)

            if os.path.isfile(mask_path):
                self.images.append(image_path)
                self.masks.append(mask_path)

    
    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE).astype('float32') / 255
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE).astype('float32') / 255

        # print(self.images[index], self.masks[index])


        # unsqueeze to add a channel dimension
        image = np.expand_dims(image, axis=2)
        mask = np.expand_dims(mask, axis=2)

        image, mask = self.transforms(image, mask)

        return image, mask


    def __len__(self):
        return len(self.images)
