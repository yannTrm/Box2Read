import os
import json
import torch
from typing import Tuple
from PIL import Image
from .base_dataset import BaseDataset 

class OdometerDataset(BaseDataset):
    """
    A PyTorch Dataset class for loading images and mileage labels for odometer reading.
    The dataset is structured in the following way:
        - `images` folder contains the images.
        - `labels` folder contains YOLO annotations in text format.
        - A JSON file with mileage labels for each image.
    
    Args:
        root_dir (str): Path to the root directory containing `train` and `val` subdirectories.
        split (str): The dataset split to use ('train' or 'val').
        labels_file (str): Path to the JSON file containing mileage labels.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
        transform (callable, optional): A transform to apply to the cropped images.
    """

    CHARS = '0123456789.'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}      
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, 
                 root_dir: str, 
                 split: str, 
                 labels_file: str, 
                 img_height: int = 32, 
                 img_width: int = 100, 
                 transform=None):
        # Appel du constructeur de la classe parente (BaseDataset)
        super().__init__(root_dir, split, labels_file, transform)
        
        self.img_height = img_height
        self.img_width = img_width

        # Lecture des labels spécifiques à l'odomètre
        with open(labels_file, 'r') as f:
            all_labels = json.load(f)
        
        self.labels = {
            file: mileage for file, mileage in all_labels.items()
            if os.path.isfile(os.path.join(self.images_dir, file)) and self._is_valid_label(mileage)
        }
        
        self.image_files = sorted(self.labels.keys())

    def _is_valid_label(self, label: str) -> bool:
        """Check if the label contains only valid characters."""
        return all(c in self.CHAR2LABEL for c in label)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetches a cropped image and its corresponding mileage label at a given index.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            Tuple: A tuple containing the cropped image as a tensor, the target sequence, and the target length.
        """
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        ann_path = os.path.join(self.labels_dir, img_file.replace('.jpg', '.txt'))
        
        image = Image.open(img_path).convert("RGB")  # Open image in RGB
        
        # Lecture de l'annotation
        with open(ann_path, 'r') as f:
            line = f.readline().strip()
            _, x_center, y_center, width, height = map(float, line.split())
        
        img_width, img_height = image.size
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image = cropped_image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)

        # Appliquer les transformations si elles existent
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        # Récupération du label pour cette image
        label = self.labels[img_file]
        target = [self.char2label(c) for c in label]
        target_length = [len(target)]

        # Conversion du label et de sa longueur en tensors
        target_tensor = torch.LongTensor(target)
        target_length_tensor = torch.LongTensor(target_length)
        
        return cropped_image, target_tensor, target_length_tensor

    def char2label(self, char: str) -> int:
        """Converts a character to its corresponding label index for odometer reading."""
        return self.CHAR2LABEL[char]
