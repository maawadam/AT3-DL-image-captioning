from torchvision import transforms
from torch.utils.data import DataLoader
from utils.caption_dataset import CaptionDataset


def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def load_split_ids(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f.readlines()]
