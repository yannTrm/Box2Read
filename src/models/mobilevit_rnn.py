import torch
import torch.nn as nn
from typing import List
import torchvision.models as models



class MobileViT_RNN(nn.Module):
    """
    MobileViT + RNN model for text recognition in images.

    Attributes:
        cnn (nn.Module): MobileViT backbone for feature extraction.
        cnn_output_size (int): Size of the features extracted after the MobileViT backbone.
        seq_len (int): Length of the sequence output after the backbone.
        map_to_seq (nn.Linear): Linear layer to map extracted features to sequences.
        rnn1 (nn.LSTM): First bidirectional LSTM layer.
        rnn2 (nn.LSTM): Second bidirectional LSTM layer.
        dense (nn.Linear): Final linear layer for classification.

    Args:
        img_channel (int): Number of channels in the input image (e.g., 1 for grayscale images).
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
        num_class (int): Number of classes to predict (e.g., number of characters to predict).
        model_size (str, optional): Model size, can be 's', 'm', 'l'. Default is 'm'.
    """

    def __init__(
        self, 
        img_channel: int, 
        img_height: int, 
        img_width: int, 
        num_class: int, 
        model_size: str = "m"
    ) -> None:
        super(MobileViT_RNN, self).__init__()

        size_configs = {
            "s": {"rnn_hidden": 128, "map_to_seq_hidden": 64},
            "m": {"rnn_hidden": 256, "map_to_seq_hidden": 128},
            "l": {"rnn_hidden": 512, "map_to_seq_hidden": 256},
        }
        config = size_configs[model_size]

        self.cnn = self._mobilevit_backbone(img_channel)

        with torch.no_grad():
            dummy_input = torch.randn(1, img_channel, img_height, img_width)
            conv_output = self.cnn(dummy_input)
            _, channel, height, width = conv_output.size()
            self.cnn_output_size = channel * height
            self.seq_len = width

        self.map_to_seq = nn.Linear(self.cnn_output_size, config["map_to_seq_hidden"])

        self.rnn1 = nn.LSTM(config["map_to_seq_hidden"], config["rnn_hidden"], bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2 * config["rnn_hidden"], config["rnn_hidden"], bidirectional=True, batch_first=True)

        self.dense = nn.Linear(2 * config["rnn_hidden"], num_class)

    def _mobilevit_backbone(self, img_channel: int) -> nn.Module:
        """
        Create a MobileViT-based backbone for feature extraction.

        Args:
            img_channel (int): Number of input channels.

        Returns:
            nn.Module: MobileViT-based feature extractor.
        """
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.features[0][0] = nn.Conv2d(
            img_channel, 
            backbone.features[0][0].out_channels, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
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

        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)

