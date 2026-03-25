"""
BEV ↔ 全景图几何接口（与 generate_pointcloud.world_to_pano_uv 数值约定一致）。

等距柱状投影与 C++ PanoramicProjector::projectImpl / undistortImpl 一致：
    lon = atan2(x_cam, z_cam)
    lat = asin(y_cam / ||P_cam||)
    col = (0.5 - lon / 2π) * W
    row = (0.5 - lat / π) * H
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch


def world_to_pano_uv_torch(
    world_xyz: torch.Tensor,
    pose_twc: torch.Tensor,
    pano_w: int,
    pano_h: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """世界坐标 → 全景图像素 (col, row)，与 generate_pointcloud.world_to_pano_uv 一致。

    Args:
        world_xyz: (B, Q, 3) 世界系点坐标
        pose_twc: (B, N, 4, 4) camera-to-world，与数据集 pose 一致
        pano_w, pano_h: 全景图宽高（像素）

    Returns:
        uv: (B, Q, N, 2) 浮点像素坐标 (col, row)
        depth: (B, Q, N) 相机系欧氏距离
        valid: (B, Q, N) bool，是否在图像内
    """
    if world_xyz.dim() != 3 or world_xyz.size(-1) != 3:
        raise ValueError(f"world_xyz must be (B, Q, 3), got {world_xyz.shape}")
    if pose_twc.dim() != 4 or pose_twc.shape[-2:] != (4, 4):
        raise ValueError(f"pose_twc must be (B, N, 4, 4), got {pose_twc.shape}")

    B, Q, _ = world_xyz.shape
    N = pose_twc.shape[1]
    device = world_xyz.device
    dtype = world_xyz.dtype

    tcw = torch.linalg.inv(pose_twc)  # (B, N, 4, 4)
    homo = torch.cat(
        [world_xyz, torch.ones(B, Q, 1, device=device, dtype=dtype)],
        dim=-1,
    )  # (B, Q, 4)

    tcw_3 = tcw[:, :, :3, :]  # (B, N, 3, 4)
    homo_e = homo.unsqueeze(1).unsqueeze(-1)  # (B, 1, Q, 4, 1)
    cam = torch.matmul(tcw_3.unsqueeze(2), homo_e).squeeze(-1)  # (B, N, Q, 3)
    cam = cam.transpose(1, 2).contiguous()  # (B, Q, N, 3)

    cx, cy, cz = cam[..., 0], cam[..., 1], cam[..., 2]
    depth = torch.sqrt(cx * cx + cy * cy + cz * cz + 1e-12)
    lon = torch.atan2(cx, cz)
    lat = torch.asin((cy / depth).clamp(-1.0, 1.0))

    pw = float(pano_w)
    ph = float(pano_h)
    col = (0.5 - lon / (2.0 * math.pi)) * pw
    row = (0.5 - lat / math.pi) * ph

    valid = (
        (col >= 0)
        & (col < pw)
        & (row >= 0)
        & (row < ph)
        & (depth > 1e-6)
    )
    uv = torch.stack([col, row], dim=-1)
    return uv, depth, valid


def bev_norm_xy_to_world_xyz(
    norm_xy: torch.Tensor,
    origin_xz: torch.Tensor,
    grid_hw: Tuple[int, int],
    meters_per_pixel: Union[float, torch.Tensor],
    world_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """归一化 BEV 坐标 [0,1]^2 → 世界 (x, y, z)。

    与 generate_bev.pixel_to_world 一致（连续近似）：
        world_x = origin_x + norm_x * W * mpp
        world_z = origin_z + norm_y * H * mpp
    其中 norm_x 对应列/X，norm_y 对应行/Z（图像向下为 Z 增大）。

    Args:
        norm_xy: (B, L, 2)，sigmoid 后 [0,1]
        origin_xz: (B, 2) 分别为 origin_x, origin_z（米）
        grid_hw: (H, W) BEV 图像高宽，与 norm 所在平面一致
        meters_per_pixel: 米/像素
        world_y: (B,) 或标量，默认 0

    Returns:
        (B, L, 3) 世界坐标
    """
    H, W = int(grid_hw[0]), int(grid_hw[1])
    B, L, _ = norm_xy.shape
    device = norm_xy.device
    dtype = norm_xy.dtype

    ox = origin_xz[:, 0:1]  # (B, 1)
    oz = origin_xz[:, 1:2]
    if isinstance(meters_per_pixel, torch.Tensor):
        mpp = meters_per_pixel.to(device=device, dtype=dtype)
        if mpp.dim() == 0:
            mpp = mpp.view(1, 1)
        else:
            mpp = mpp.view(B, 1)
    else:
        mpp = torch.full((B, 1), float(meters_per_pixel), device=device, dtype=dtype)

    wx = ox.unsqueeze(1) + norm_xy[..., 0:1] * (float(W) * mpp.unsqueeze(1))
    wz = oz.unsqueeze(1) + norm_xy[..., 1:2] * (float(H) * mpp.unsqueeze(1))

    if world_y is None:
        wy = torch.zeros(B, L, 1, device=device, dtype=dtype)
    else:
        if world_y.dim() == 0:
            wy = world_y.view(1, 1, 1).expand(B, L, 1)
        else:
            wy = world_y.view(B, 1, 1).expand(B, L, 1)
    return torch.cat([wx, wy, wz], dim=-1)


def _infer_query_points(bev_points=None, bev_regions=None):
    """Normalize BEV query inputs to a `(B, Q, 2)` tensor."""
    if bev_points is not None:
        if bev_points.ndim != 3 or bev_points.shape[-1] != 2:
            raise ValueError("bev_points must have shape (B, Q, 2)")
        return bev_points

    if bev_regions is None:
        raise ValueError("Either bev_points or bev_regions must be provided")

    if bev_regions.ndim == 3 and bev_regions.shape[-1] == 4:
        x0, y0, x1, y1 = torch.unbind(bev_regions, dim=-1)
        return torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5), dim=-1)

    if bev_regions.ndim == 4 and bev_regions.shape[-1] == 2:
        return bev_regions.float().mean(dim=-2)

    raise ValueError(
        "bev_regions must have shape (B, Q, 4) or (B, Q, K, 2)"
    )


def _build_candidate_mask(query_points, pano_counts, num_panos):
    batch_size, num_queries = query_points.shape[:2]
    device = query_points.device
    candidate_mask = torch.zeros(
        batch_size, num_queries, num_panos, dtype=torch.bool, device=device
    )

    for b in range(batch_size):
        valid = int(pano_counts[b].item())
        if valid > 0:
            candidate_mask[b, :, :valid] = True

    return candidate_mask


def project_to_bev(
    pose_matrices,
    pano_counts,
    bev_points=None,
    bev_regions=None,
    world_query_xyz=None,
    bev_world_meta=None,
    depth_images=None,
    bev_size=None,
    pano_image_size=None,
):
    """BEV 查询点 → 各全景上的反投影 UV（训练/调试与 generate_pointcloud 一致）。

    在下列输入之一时计算真实 ``projected_uv`` / ``visibility_mask``：
        - ``world_query_xyz``: (B, Q, 3)
        - 或 ``bev_points`` (B, Q, 2) 归一化 [0,1] + ``bev_world_meta``

    ``bev_world_meta`` 应为 dict（tensor 已在正确 device/dtype）::
        - origin_xz: (B, 2)
        - grid_hw: (H, W) 元组，与 BEV 张量高宽一致
        - meters_per_pixel: float 或 (B,)
        - world_y: 可选 (B,) 或标量 tensor

    Args:
        pose_matrices: (B, N, 4, 4)
        pano_counts: (B,)
        pano_image_size: (H_p, W_p)，未提供时无法用几何，仅返回占位

    Returns:
        dict 含 query_points, candidate_pano_mask, projected_uv (B,Q,N,2),
        visibility_mask (B,Q,N), debug_info
    """
    if pose_matrices is None or pano_counts is None:
        raise ValueError("pose_matrices and pano_counts are required")

    if world_query_xyz is None:
        query_points = _infer_query_points(bev_points=bev_points, bev_regions=bev_regions)
        if bev_world_meta is not None:
            world_query_xyz = bev_norm_xy_to_world_xyz(
                query_points,
                bev_world_meta["origin_xz"],
                tuple(bev_world_meta["grid_hw"]),
                bev_world_meta.get("meters_per_pixel", 0.02),
                bev_world_meta.get("world_y"),
            )
    else:
        B, Q, _ = world_query_xyz.shape
        if bev_points is not None:
            query_points = bev_points
        else:
            query_points = torch.zeros(B, Q, 2, device=world_query_xyz.device, dtype=world_query_xyz.dtype)

    batch_size, num_queries = query_points.shape[:2]
    num_panos = pose_matrices.shape[1]
    device = pose_matrices.device
    dtype = pose_matrices.dtype

    candidate_pano_mask = _build_candidate_mask(query_points, pano_counts, num_panos)

    projected_uv = torch.full(
        (batch_size, num_queries, num_panos, 2),
        -1.0,
        dtype=dtype,
        device=device,
    )
    visibility_mask = torch.zeros(
        batch_size, num_queries, num_panos, dtype=torch.bool, device=device
    )

    implemented = False
    if world_query_xyz is not None and pano_image_size is not None:
        ph, pw = int(pano_image_size[0]), int(pano_image_size[1])
        uv, _depth, valid = world_to_pano_uv_torch(
            world_query_xyz.to(dtype=dtype),
            pose_matrices,
            pw,
            ph,
        )
        projected_uv = uv
        visibility_mask = valid & candidate_pano_mask
        implemented = True

    return {
        "query_points": query_points,
        "candidate_pano_mask": candidate_pano_mask,
        "projected_uv": projected_uv,
        "visibility_mask": visibility_mask,
        "world_query_xyz": world_query_xyz,
        "debug_info": {
            "implemented": implemented,
            "has_depth": depth_images is not None,
            "bev_size": bev_size,
            "pano_image_size": pano_image_size,
            "selection_strategy": "equirectangular_backproject",
        },
    }


def build_pano_attn_bias_from_uv(
    projected_uv: torch.Tensor,
    token_uv: torch.Tensor,
    pano_counts: torch.Tensor,
    tokens_per_pano: int,
    sigma_px: float = 64.0,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """根据反投影 UV 与 token 像素位置构造加性注意力偏置 (B, L, S)。

    S = N_max * tokens_per_pano；token 顺序与 PanoEncoder flatten 一致：
    先 pano n=0 的 tpp 个 token，再 n=1 ...

    Args:
        projected_uv: (B, L, N_max, 2) 列为 col、行为 row（像素）
        token_uv: (B, S, 2) 每个 key 在全景输入分辨率下的 (col, row)
        pano_counts: (B,) 有效全景数
        sigma_px: 高斯核宽度（像素）
        key_padding_mask: (B, S) True 表示 padding，对应偏置为 -inf

    Returns:
        attn_bias: (B, L, S)，传给 scaled_dot_product_attention 的 attn_mask
    """
    B, L, N_max, _ = projected_uv.shape
    S = token_uv.shape[1]
    device = projected_uv.device
    dtype = projected_uv.dtype

    n_per_s = (torch.arange(S, device=device) // tokens_per_pano).long()
    proj_n = projected_uv[:, :, n_per_s, :]

    diff = proj_n - token_uv.unsqueeze(1)  # (B, L, S, 2)
    dist_sq = (diff ** 2).sum(dim=-1)  # (B, L, S)
    bias = -dist_sq / (2.0 * sigma_px * sigma_px)

    valid_n = (n_per_s.view(1, S) < pano_counts.view(B, 1)).unsqueeze(1).expand(B, L, S)
    bias = bias.masked_fill(~valid_n, 0.0)

    if key_padding_mask is not None:
        bias = bias.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))

    return bias.to(dtype=dtype)
