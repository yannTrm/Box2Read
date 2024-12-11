import torch
import torch.nn as nn
from typing import Tuple

class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings for the Vision Transformer encoder.
    """
    def __init__(self, img_size: Tuple[int, int], patch_size: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = nn.Conv2d(
            in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.projection(x)  # Shape: (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)

        # Add [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # Shape: (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embedding
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer Encoder for TrOCR.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_dim: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # Shape: (B, seq_len, embed_dim)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for TrOCR.
    """
    def __init__(
        self, embed_dim: int, num_heads: int, num_layers: int, mlp_dim: int, vocab_size: int, dropout: float
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt_seq_len = tgt.size(1)
        tgt = self.embedding(tgt) + self.pos_embedding[:, :tgt_seq_len, :]
        tgt = self.decoder(tgt, memory)
        output = self.output_layer(tgt)
        return output


class TrOCR(nn.Module):
    """
    TrOCR: Transformer OCR model combining a Vision Transformer encoder and a Transformer decoder.
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 100),
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        mlp_dim: int = 512,
        vocab_size: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim)
        self.encoder = VisionTransformerEncoder(embed_dim, num_heads, num_encoder_layers, mlp_dim, dropout)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_decoder_layers, mlp_dim, vocab_size, dropout)

    def forward(self, images: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.patch_embedding(images)  # Shape: (B, num_patches + 1, embed_dim)
        memory = self.encoder(memory)  # Shape: (B, num_patches + 1, embed_dim)
        output = self.decoder(tgt, memory)  # Shape: (B, tgt_len, vocab_size)
        return output

