import torch
import torch.nn as nn
from typing import Tuple

class CRNN(nn.Module):
    def __init__(self, img_channel: int, img_height: int, img_width: int, num_class: int,
                 map_to_seq_hidden: int = 64, rnn_hidden: int = 256, leaky_relu: bool = False) -> None:
        super(CRNN, self).__init__()

        self.cnn, (output_channel, output_height, output_width) = self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def _cnn_backbone(self, img_channel: int, img_height: int, img_width: int, leaky_relu: bool) -> Tuple[nn.Sequential, Tuple[int, int, int]]:
        assert img_height % 16 == 0

        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i: int, batch_norm: bool = False) -> None:
            input_channel = channels[i]
            output_channel = channels[i + 1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        conv_relu(2)
        conv_relu(3)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=(2, 1)))
        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))
        conv_relu(6)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)