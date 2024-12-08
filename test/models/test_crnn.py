import torch

import sys
sys.path.append(".")

from src.models.crnn import CRNN


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(img_channel=1, img_height=32, img_width=100, num_class=10).to(device)

# Create a dummy input tensor with a different width
dummy_input = torch.randn(2, 1, 32, 200).to(device)  # (batch_size, channels, height, width)
print(f"Dummy input shape: {dummy_input.shape}")

# Pass the dummy input through the model
dummy_output = model(dummy_input)
print(f"Dummy output shape: {dummy_output.shape}")