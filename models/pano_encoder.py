"""Panorama feature encoder for RoomFormer.

Extracts visual features from N panorama images per scene and produces
d_model-dimensional tokens with pose-based positional encoding, ready
for cross-attention in the Deformable Transformer decoder.
"""

import torch
import torch.nn as nn
import torchvision


class PanoEncoder(nn.Module):
    """Encode multiple panorama images into a flat token sequence."""

    def __init__(self, d_model=256, backbone_name='resnet18',
                 freeze_backbone=True, use_gradient_checkpointing=False):
        super().__init__()

        resnet = getattr(torchvision.models, backbone_name)(pretrained=True)
        self.body = nn.Sequential(*list(resnet.children())[:-2])

        feat_channels = 512 if backbone_name in ('resnet18', 'resnet34') else 2048
        self.proj = nn.Sequential(
            nn.Conv2d(feat_channels, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )

        self.pose_embed = nn.Sequential(
            nn.Linear(12, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing

        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad_(False)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.pose_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _extract_features(self, x):
        return self.proj(self.body(x))

    def forward(self, pano_images, pose_matrices, pano_counts):
        """
        Args:
            pano_images:   (B, N_max, 3, Hp, Wp)  zero-padded panorama batch
            pose_matrices: (B, N_max, 4, 4)        zero-padded pose matrices
            pano_counts:   (B,)  actual panorama count per sample

        Returns:
            pano_tokens: (B, T, d_model)   T = N_max * h' * w'
            pano_pos:    (B, T, d_model)   pose-based positional encoding
            pano_mask:   (B, T)            True = padded (for key_padding_mask)
        """
        B, N_max, C, Hp, Wp = pano_images.shape
        device = pano_images.device

        x = pano_images.reshape(B * N_max, C, Hp, Wp)

        if self.use_gradient_checkpointing and self.training:
            feat = torch.utils.checkpoint.checkpoint(
                self._extract_features, x, use_reentrant=False)
        else:
            feat = self._extract_features(x)

        _, d, h, w = feat.shape
        tpp = h * w  # tokens per panorama

        tokens = feat.flatten(2).transpose(1, 2).reshape(B, N_max, tpp, d)

        pose_flat = pose_matrices[:, :, :3, :].reshape(B, N_max, 12)
        pose_feat = self.pose_embed(pose_flat)
        pose_feat = pose_feat.unsqueeze(2).expand(-1, -1, tpp, -1)

        T = N_max * tpp
        tokens_flat = tokens.reshape(B, T, d)
        pos_flat = pose_feat.reshape(B, T, d)

        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            n = int(pano_counts[b].item())
            mask[b, :n * tpp] = False

        return tokens_flat, pos_flat, mask


def build_pano_encoder(args):
    if not getattr(args, 'use_pano', False):
        return None
    return PanoEncoder(
        d_model=args.hidden_dim,
        backbone_name=getattr(args, 'pano_backbone', 'resnet18'),
        freeze_backbone=getattr(args, 'freeze_pano_backbone', True),
        use_gradient_checkpointing=getattr(args, 'pano_grad_ckpt', False),
    )
