import torch
import sys
sys.path.append(".")

#from src.models.mobilevit_rnn import MobileViT_RNN

import torch
import torch.nn as nn
from typing import List
import torchvision.models as models


class MobileViT_RNN(nn.Module):
    def __init__(
        self, 
        img_channel: int, 
        img_height: int, 
        img_width: int, 
        num_class: int, 
        model_size: str = "m",
        max_seq_len: int = 100  # Ajout du paramètre max_seq_len
    ) -> None:
        super(MobileViT_RNN, self).__init__()

        size_configs = {
            "s": {"rnn_hidden": 128, "map_to_seq_hidden": 64},
            "m": {"rnn_hidden": 256, "map_to_seq_hidden": 128},
            "l": {"rnn_hidden": 512, "map_to_seq_hidden": 256},
        }
        config = size_configs[model_size]

        self.cnn = self._mobilevit_backbone(img_channel)
        self.max_seq_len = max_seq_len  # Stocker le paramètre

        with torch.no_grad():
            dummy_input = torch.randn(1, img_channel, img_height, img_width)
            conv_output = self.cnn(dummy_input)
            _, channel, height, width = conv_output.size()
            self.cnn_output_size = channel * height
            self.seq_len = min(width, self.max_seq_len)  # Utiliser max_seq_len pour limiter la longueur de la séquence

        self.map_to_seq = nn.Linear(self.cnn_output_size, config["map_to_seq_hidden"])

        self.rnn1 = nn.LSTM(config["map_to_seq_hidden"], config["rnn_hidden"], bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2 * config["rnn_hidden"], config["rnn_hidden"], bidirectional=True, batch_first=True)

        self.dense = nn.Linear(2 * config["rnn_hidden"], num_class)

    def _mobilevit_backbone(self, img_channel: int) -> nn.Module:
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.features[0][0] = nn.Conv2d(
            img_channel, 
            backbone.features[0][0].out_channels, 
            kernel_size=3, 
            stride=1,  # Changer le stride de 2 à 1 pour réduire moins la dimension spatiale
            padding=1
        )
        # Réduire le nombre de couches de pooling
        for i in range(1, len(backbone.features)):
            if isinstance(backbone.features[i], nn.MaxPool2d):
                backbone.features[i] = nn.Identity()  # Remplacer les couches de pooling par des identités
            elif isinstance(backbone.features[i], nn.Conv2d):
                backbone.features[i].stride = (1, 1)  # Réduire les strides des convolutions

        backbone.classifier = nn.Identity()  # Remove the classifier head
        return backbone.features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pass the input images through the network to obtain the output.

        Args:
            images (torch.Tensor): Input tensor of images of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch, num_class).
        """
        conv = self.cnn(images)  # (batch, channel, height, width)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)  # (batch, feature, seq_len)
        conv = conv.permute(2, 0, 1)  # (seq_len, batch, feature)
        print(f"conv {conv.shape}")


        seq = self.map_to_seq(conv)
        print(f"seq {seq.shape}")


        recurrent, _ = self.rnn1(seq)
        print(f"rec1 {recurrent.shape}")
        recurrent, _ = self.rnn2(recurrent)
        print(f"rec2 {recurrent.shape}")

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

def model_size_in_mb(model: torch.nn.Module) -> float:
    """Calculer la taille du modèle en Mo."""
    param_size = sum(p.numel() for p in model.parameters()) * 4  # size  bytes, for float32 (4 bytes)
    return param_size / (1024 ** 2) 

# Test avec différentes tailles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for size in ["s", "m", "l"]:
    model = MobileViT_RNN(img_channel=1, img_height=32, img_width=200, num_class=10, model_size=size).to(device)
    dummy_input = torch.randn(2, 1, 32, 200).to(device)
    dummy_output = model(dummy_input)
    model_size = model_size_in_mb(model)
    print(f"Model size: {size}, Output shape: {dummy_output.shape}, Model size: {model_size:.2f} MB")
