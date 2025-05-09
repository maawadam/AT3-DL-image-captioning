import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_folder,
        captions_tensor,
        lengths_tensor,
        image_filenames,
        transform=None,
    ):
        self.image_folder = image_folder
        self.captions = captions_tensor
        self.lengths = lengths_tensor
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.captions[idx], self.lengths[idx]
