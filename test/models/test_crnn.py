import torch
import sys
sys.path.append(".")

from src.models.crnn import CRNN



def model_size_in_mb(model: torch.nn.Module) -> float:
    """Calculer la taille du modèle en Mo."""
    param_size = sum(p.numel() for p in model.parameters()) * 4  # size  bytes, for float32 (4 bytes)
    return param_size / (1024 ** 2) 

# Test avec différentes tailles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for size in ["n","n-bis", "s", "m", "l", "xl"]:
    model = CRNN(img_channel=1, img_height=32, img_width=100, num_class=10,  model_size=size).to(device)
    dummy_input = torch.randn(2, 1, 32, 100).to(device)
    dummy_output = model(dummy_input)
    model_size = model_size_in_mb(model)
    print(f"Model size: {size}, Output shape: {dummy_output.shape}, Model size: {model_size:.2f} MB")
