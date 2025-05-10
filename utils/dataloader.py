import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from utils.caption_dataset import CaptionDataset


# Load split image IDs
def load_split_ids(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f.readlines()]


# Default transforms
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


def build_caption_dataset(
    image_ids, image_caption_seqs, word2idx, image_folder, transform
):
    image_filenames = []
    caption_tensors = []
    lengths = []

    for img_id in image_ids:
        if img_id in image_caption_seqs:
            for seq in image_caption_seqs[img_id]:
                image_filenames.append(img_id)
                caption_tensors.append(torch.tensor(seq))
                lengths.append(len(seq))

    padded_seqs = pad_sequence(
        caption_tensors, batch_first=True, padding_value=word2idx["<pad>"]
    )
    lengths_tensor = torch.tensor(lengths)

    dataset = CaptionDataset(
        image_folder=image_folder,
        captions_tensor=padded_seqs,
        lengths_tensor=lengths_tensor,
        image_filenames=image_filenames,
        transform=transform,
    )
    return dataset
