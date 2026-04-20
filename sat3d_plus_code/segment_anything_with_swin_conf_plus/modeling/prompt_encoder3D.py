# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Optional, Tuple, Type, List
from transformers import AutoTokenizer, AutoModel


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class PromptEncoder3D(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        freeze_text_encoder: bool = True,
        max_text_length: int = 32,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder, now including a pretrained
        biomedical text encoder for free-form medical descriptions.

        Arguments:
          in_chans (int): number of input channels for mask/conf prompt encoding
          embed_dim (int): the prompt embedding dimension
          image_embedding_size (tuple): spatial size of the image embedding (D, H, W)
          input_image_size (tuple): padded size of the input image (D, H, W)
          mask_in_chans (int): number of hidden channels for mask encoding
          activation (nn.Module): activation to use
          text_model_name (str): HuggingFace model name for text encoder
          freeze_text_encoder (bool): whether to freeze the text encoder
          max_text_length (int): maximum token length for text
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom3D(embed_dim // 3)
        self.max_text_length = max_text_length

        # 4 embeddings:
        # 0: negative point, 1: positive point, 2: box min, 3: box max
        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            image_embedding_size[0],
            image_embedding_size[1],
            image_embedding_size[2],
        )

        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(in_chans, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans // 4),
            activation(),
            nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            nn.Conv3d(mask_in_chans, embed_dim // 2, kernel_size=1),
        )

        self.conf_downscaling = nn.Sequential(
            nn.Conv3d(in_chans, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans // 4),
            activation(),
            nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            nn.Conv3d(mask_in_chans, embed_dim // 2, kernel_size=1),
        )

        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # ---------- Text encoder ----------
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, embed_dim)
        # ----------------------------------

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points with the shape of the image encoding.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1 x C x D x H x W

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # shift to center of voxel

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        assert (labels >= -1).all() and (labels <= 1).all(), (
            f"Labels out of range: {labels.unique()}"
        )

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        device = point_embedding.device
        not_a_point_weight = self.not_a_point_embed.weight.to(device)
        neg_point_weight = self.point_embeddings[0].weight.to(device)
        pos_point_weight = self.point_embeddings[1].weight.to(device)

        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += not_a_point_weight
        point_embedding[labels == 0] += neg_point_weight
        point_embedding[labels == 1] += pos_point_weight

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embeds 3D box prompts.

        boxes: (B, 2, 3) where the two corners are (min, max)
        """
        boxes = boxes + 0.5  # shift to center of voxel
        corner_embedding = self.pe_layer.forward_with_coords(
            boxes, self.input_image_size
        )  # (B, 2, C)

        device = corner_embedding.device
        min_corner_weight = self.point_embeddings[2].weight.to(device)
        max_corner_weight = self.point_embeddings[3].weight.to(device)

        corner_embedding[:, 0, :] += min_corner_weight
        corner_embedding[:, 1, :] += max_corner_weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Embeds mask and confidence inputs."""
        mask_embedding = self.mask_downscaling(masks)
        conf_embedding = self.conf_downscaling(conf)
        dense_embeddings = torch.cat([mask_embedding, conf_embedding], dim=1)
        return dense_embeddings

    def _embed_text(self, text: List[str], device: torch.device) -> torch.Tensor:
        """
        Embeds free-form text using PubMedBERT with attention-mask-aware mean pooling.

        Returns:
            text_emb: (B, 1, embed_dim)
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_encoder(**tokens)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        attention_mask = tokens["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # (B, L, 1)
        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)  # (B, H)
        valid_token_count = attention_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

        text_feat = sum_hidden / valid_token_count  # (B, H)
        text_emb = self.text_proj(text_feat)        # (B, embed_dim)
        text_emb = text_emb.unsqueeze(1)            # (B, 1, embed_dim)

        return text_emb

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text: Optional[List[str]] = None,
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        if text is not None:
            return len(text)
        return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        conf: Optional[torch.Tensor],
        text: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Arguments:
          points: tuple(coords, labels)
              coords: (B, N, 3)
              labels: (B, N)
          boxes:
              (B, 2, 3)
          masks:
              (B, 1, D, H, W)
          conf:
              (B, 1, D, H, W)
          text:
              list of strings, length B

        Returns:
          sparse_embeddings: (B, total_prompts, embed_dim)
          dense_embeddings: (B, embed_dim, embed_D, embed_H, embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, text)
        device = self._get_device()

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if text is not None:
            text_embeddings = self._embed_text(text, device)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            if conf is None:
                raise ValueError("conf must be provided when masks is not None.")
            dense_embeddings = self._embed_masks(masks, conf)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
                self.image_embedding_size[2],
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom3D(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points normalized to [0, 1]."""
        coords = 2 * coords - 1
        gaussian_matrix = self.positional_encoding_gaussian_matrix.to(coords.device)
        coords = coords @ gaussian_matrix
        coords = 2 * np.pi * coords

        # Keeps same behavior as your existing version
        return torch.cat(
            [torch.sin(coords), torch.cos(coords), torch.sin(coords)],
            dim=-1,
        )

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        x, y, z = size
        device = self.positional_encoding_gaussian_matrix.device

        grid = torch.ones((x, y, z), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5

        y_embed = y_embed / y
        x_embed = x_embed / x
        z_embed = z_embed / z

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x D x H x W

    def forward_with_coords(
        self,
        coords_input: torch.Tensor,
        image_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0, 1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C