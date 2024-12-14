import torch
import torch.nn as nn
from typing import List, Tuple


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for text recognition in images.

    Attributes:
        cnn (nn.Sequential): CNN backbone for feature extraction.
        cnn_output_size (int): Size of the features extracted after passing through the CNN.
        seq_len (int): Length of the sequence output after the CNN.
        map_to_seq (nn.Linear): Linear layer to map extracted features to sequences.
        rnn1 (nn.LSTM): First bidirectional LSTM layer.
        rnn2 (nn.LSTM): Second bidirectional LSTM layer.
        dense (nn.Linear): Final linear layer for classification.

    Args:
        img_channel (int): Number of channels in the input image (e.g., 1 for grayscale images).
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
        num_class (int): Number of classes to predict (e.g., number of characters to predict).
        model_size (str, optional): Model size, can be 'n', 's', 'm', 'l', 'xl'. Default is 'm'.
        leaky_relu (bool, optional): Whether to use LeakyReLU as the activation function instead of ReLU. Default is False.
    """

    def __init__(
        self,
        img_channel: int,
        img_height: int,
        img_width: int,
        num_class: int,
        model_size: str = "m",
        leaky_relu: bool = False,
    ) -> None:
        """
        Initialize the CRNN model.

        Args:
            img_channel (int): Number of channels in the input image.
            img_height (int): Height of the input image.
            img_width (int): Width of the input image.
            num_class (int): Number of output classes.
            model_size (str, optional): Model size ('n', 's', 'm', 'l', 'xl').
            leaky_relu (bool, optional): Whether to use LeakyReLU or ReLU.
        """

        super(CRNN, self).__init__()

        size_configs = {
            "n": {"cnn_channels": [img_channel, 32, 64, 128], "rnn_hidden": 128, "map_to_seq_hidden": 32},
            "n-bis": {"cnn_channels": [img_channel, 32, 64, 128], "rnn_hidden": 128, "map_to_seq_hidden": 128},
            "s": {"cnn_channels": [img_channel, 64, 128, 256], "rnn_hidden": 256, "map_to_seq_hidden": 64},
            "m": {"cnn_channels": [img_channel, 64, 128, 256, 256], "rnn_hidden": 256, "map_to_seq_hidden": 64},
            "l": {"cnn_channels": [img_channel, 64, 128, 256, 256, 512], "rnn_hidden": 512, "map_to_seq_hidden": 128},
            "xl": {"cnn_channels": [img_channel, 64, 128, 256, 256, 512, 512], "rnn_hidden": 512, "map_to_seq_hidden": 128},
        }
        config = size_configs[model_size]

        self.cnn = self._cnn_backbone(config["cnn_channels"], leaky_relu)

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

    def _cnn_backbone(self, cnn_channels: List[int], leaky_relu: bool) -> nn.Sequential:
        """
        Creates the CNN backbone with convolutional layers and activations.

        Args:
            cnn_channels (List[int]): List of channels for each layer of the CNN.
            leaky_relu (bool): Whether to use LeakyReLU or ReLU.

        Returns:
            nn.Sequential: Complete CNN backbone with convolution, activation, and pooling layers.
        """

        cnn = nn.Sequential()

        def conv_relu(i: int, batch_norm: bool = False) -> None:
            """
            Fonction auxiliaire pour ajouter une couche de convolution suivie d'une activation.

            Args:
                i (int): Indice de la couche.
                batch_norm (bool): Activer la normalisation par lot.
            """
            input_channel = cnn_channels[i]
            output_channel = cnn_channels[i + 1]
            cnn.add_module(
                f"conv{i}",
                nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            )
            if batch_norm:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(output_channel))
            activation = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f"relu{i}", activation)

        for i in range(len(cnn_channels) - 1):
            conv_relu(i, batch_norm=i >= 2)
            if i % 2 == 1:  
                cnn.add_module(f"pooling{i // 2}", nn.MaxPool2d(kernel_size=2, stride=2))

        return cnn

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pass the input images through the network to obtain the output.

        Args:
            images (torch.Tensor): Input tensor of images of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, num_class).
        """

        conv = self.cnn(images)  # (batch, channel, height, width)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)  # (batch, feature, seq_len)
        conv = conv.permute(2, 0, 1)  # (seq_len, batch, feature)

        assert conv.size(2) == self.cnn_output_size, (
            f"Mismatch in input features for Linear. Expected {self.cnn_output_size}, got {conv.size(2)}."
        )

        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)