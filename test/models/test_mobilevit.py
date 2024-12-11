import torch
import sys
sys.path.append(".")

from src.models.mobilevit_rnn import MobileViT_RNN


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
