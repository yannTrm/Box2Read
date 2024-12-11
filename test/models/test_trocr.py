import sys

sys.path.append(".")


import torch
import torch.nn as nn
from src.models.trocr import TrOCR

# Définir le modèle TrOCR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrOCR(img_size=(32, 100), patch_size=16, embed_dim=256, num_heads=8, vocab_size=1000).to(device)

# Définir les données factices (dummy data)
batch_size = 2
seq_len = 10
vocab_size = 1000

dummy_input = torch.randn(batch_size, 3, 32, 100).to(device)  # Batch de 2 images (3 canaux, 32x100)
dummy_tgt = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)  # Séquence de 10 tokens pour 2 exemples

# Définir la fonction de perte CTC
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# Passe avant (forward)
model.train()
output = model(dummy_input, dummy_tgt)  # Shape: (batch_size, seq_len, vocab_size)

# Convertir les sorties pour la perte CTC
output = output.log_softmax(2)  # Appliquer log_softmax sur la dimension vocab_size
output = output.permute(1, 0, 2)  # CTC Loss attend (seq_len, batch_size, vocab_size)

# Définir les longueurs des séquences d'entrée et de sortie
input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(device)
target_lengths = torch.randint(1, seq_len + 1, (batch_size,), dtype=torch.long).to(device)

# Calculer la perte
loss = ctc_loss(output, dummy_tgt, input_lengths, target_lengths)

# Passe arrière (backward)
loss.backward()

print(f"Output shape: {output.shape}")
print(f"Loss: {loss.item()}")





"""
# Tester avec un dummy input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrOCR(img_size=(32, 100), patch_size=16, embed_dim=256, num_heads=8, vocab_size=100).to(device)
dummy_input = torch.randn(2, 3, 32, 100).to(device)  # Batch de 2 images (3 canaux, 32x100)
dummy_tgt = torch.randint(0, 100, (2, 10)).to(device)  # Séquence de 10 tokens pour 2 exemples

output = model(dummy_input, dummy_tgt)  # Shape: (2, 10, vocab_size)
print("Output shape:", output.shape)
"""