import os
import json
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    """
    A generic PyTorch Dataset class for loading images and labels for various OCR tasks.
    The dataset is structured with separate folders for images and labels, 
    and a JSON file containing the labels for each image.
    
    Args:
        root_dir (str): Path to the root directory containing `train` and `val` subdirectories.
        split (str): The dataset split to use ('train' or 'val').
        labels_file (str): Path to the JSON file containing labels for each image.
        transform (callable, optional): A transform to apply to the images.
    """

    def __init__(self, 
                 root_dir: str, 
                 split: str, 
                 labels_file: str, 
                 transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.labels_dir):
            raise ValueError(f"Either 'images' or 'labels' folder is missing in {split} ({root_dir})")

        with open(labels_file, 'r') as f:
            all_labels = json.load(f)

        self.labels = {
            file: label for file, label in all_labels.items()
            if os.path.isfile(os.path.join(self.images_dir, file))
        }

        self.image_files = sorted(self.labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetches an image and its corresponding label at a given index.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            Tuple: A tuple containing the image tensor, the target sequence, and the target length.
        """
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        label = self.labels[img_file]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        target = [self.char2label(c) for c in label]
        target_length = [len(target)]

        target_tensor = torch.LongTensor(target)
        target_length_tensor = torch.LongTensor(target_length)

        return image, target_tensor, target_length_tensor

    def char2label(self, char: str) -> int:
        """Converts a character to its corresponding label index."""
        raise NotImplementedError("This method should be implemented in the subclass.")

def base_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for BaseDataset.
    
    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): List of tuples containing images, targets, and target lengths.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batched images, targets, and target lengths.
    """
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
