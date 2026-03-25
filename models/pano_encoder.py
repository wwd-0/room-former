"""Panorama feature encoder for RoomFormer.

Extracts visual features from N panorama images (RGB + optional Depth)
per scene and produces d_model-dimensional tokens with pose-based
positional encoding, ready for cross-attention in the decoder.
"""

import torch
import torch.nn as nn
import torchvision

from util.pano_projector import project_to_bev as project_bev_queries_to_panos


class PanoEncoder(nn.Module):
    """Encode multiple panorama images into a flat token sequence."""

    def __init__(self, d_model=256, backbone_name='resnet18',
                 freeze_backbone=True, use_gradient_checkpointing=False,
                 use_depth=False):
        super().__init__()

        self.use_depth = use_depth
        in_channels = 4 if use_depth else 3

        resnet = getattr(torchvision.models, backbone_name)(pretrained=True)

        if in_channels != 3:
            old_conv1 = resnet.conv1
            new_conv1 = nn.Conv2d(in_channels, old_conv1.out_channels,
                                  kernel_size=old_conv1.kernel_size,
                                  stride=old_conv1.stride,
                                  padding=old_conv1.padding,
                                  bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3] = old_conv1.weight
                new_conv1.weight[:, 3:] = 0.0
            resnet.conv1 = new_conv1

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

    def project_to_bev(self, pose_matrices, pano_counts, bev_points=None,
                       bev_regions=None, world_query_xyz=None, bev_world_meta=None,
                       depth_images=None, bev_size=None, pano_image_size=None):
        """Public wrapper for BEV-to-panorama candidate lookup / 反投影 UV。"""
        return project_bev_queries_to_panos(
            pose_matrices=pose_matrices,
            pano_counts=pano_counts,
            bev_points=bev_points,
            bev_regions=bev_regions,
            world_query_xyz=world_query_xyz,
            bev_world_meta=bev_world_meta,
            depth_images=depth_images,
            bev_size=bev_size,
            pano_image_size=pano_image_size,
        )

    def forward(self, pano_images, pose_matrices, pano_counts,
                depth_images=None):
        """
        Args:
            pano_images:   (B, N_max, 3, Hp, Wp)  zero-padded RGB
            pose_matrices: (B, N_max, 4, 4)        zero-padded poses
            pano_counts:   (B,)  actual count per sample
            depth_images:  (B, N_max, 1, Hp, Wp)  zero-padded depth (optional)

        Returns:
            pano_tokens: (B, T, d_model)
            pano_pos:    (B, T, d_model)
            pano_mask:   (B, T)  True = padded
            pano_token_uv: (B, T, 2) 各 token 在全景输入分辨率下的像素 (col, row)，与反投影一致
        """
        B, N_max, C, Hp, Wp = pano_images.shape
        device = pano_images.device
        dtype = pano_images.dtype

        if self.use_depth and depth_images is not None:
            x = torch.cat([pano_images, depth_images], dim=2)  # (B, N, 4, Hp, Wp)
        else:
            x = pano_images

        x = x.reshape(B * N_max, x.shape[2], Hp, Wp)

        if self.use_gradient_checkpointing and self.training:
            feat = torch.utils.checkpoint.checkpoint(
                self._extract_features, x, use_reentrant=False)
        else:
            feat = self._extract_features(x)

        _, d, h, w = feat.shape
        tpp = h * w

        tokens = feat.flatten(2).transpose(1, 2).reshape(B, N_max, tpp, d)

        pose_flat = pose_matrices[:, :, :3, :].reshape(B, N_max, 12)
        pose_feat = self.pose_embed(pose_flat)
        pose_feat = pose_feat.unsqueeze(2).expand(-1, -1, tpp, -1)

        cols = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / float(w) * float(Wp)
        rows = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / float(h) * float(Hp)
        try:
            grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        except TypeError:
            grid_r, grid_c = torch.meshgrid(rows, cols)
        uv_local = torch.stack([grid_c, grid_r], dim=-1).reshape(1, 1, tpp, 2)
        token_uv = uv_local.expand(B, N_max, tpp, 2).reshape(B, N_max * tpp, 2).to(dtype=dtype)

        T = N_max * tpp
        tokens_flat = tokens.reshape(B, T, d)
        pos_flat = pose_feat.reshape(B, T, d)

        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            n = int(pano_counts[b].item())
            mask[b, :n * tpp] = False

        return tokens_flat, pos_flat, mask, token_uv


def build_pano_encoder(args):
    if not getattr(args, 'use_pano', False):
        return None
    return PanoEncoder(
        d_model=args.hidden_dim,
        backbone_name=getattr(args, 'pano_backbone', 'resnet18'),
        freeze_backbone=getattr(args, 'freeze_pano_backbone', True),
        use_gradient_checkpointing=getattr(args, 'pano_grad_ckpt', False),
        use_depth=getattr(args, 'use_depth', False),
    )
