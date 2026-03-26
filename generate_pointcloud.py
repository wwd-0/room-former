#!/usr/bin/env python3
"""
从全景深度图 + 置信度图生成单点点云（相机坐标系），并提供 BEV↔全景图 的双向投影接口。

单点点云输出的是相机局部坐标，不涉及位姿变换（与 C++ generatePointCloud 一致）。
位姿变换由外部流程（如 generate_bev.py）负责。

等距柱状投影 (equirectangular) 约定与 C++ PanoramicProjector 一致：
    row ∈ [0, H)  对应纬度 lat = (0.5 - row/H) * π
    col ∈ [0, W)  对应经度 lon = (0.5 - col/W) * 2π
    方向向量：x = cos(lat)*sin(lon),  y = sin(lat),  z = cos(lat)*cos(lon)

深度缩放：raw_uint16 * depth_parm * 1e-4  (depth_parm 从 depthConfig.json 读取，默认 1.75)

目录结构：
  scene_root/
    images/{name}.jpg                     # 全景 RGB
    images_depth/{name}_depth.png         # 深度图 (uint16 PNG)
    images_depth/{name}_confidence.png    # 置信度图 (uint8 灰度)
    images_depth/depthConfig.json         # {"depth_parm": 1.75}（可选）

用法：
  python generate_pointcloud.py /path/to/scene_root
  python generate_pointcloud.py /path/to/scene_root -o /tmp/output_plys/
  python generate_pointcloud.py /path/to/scene_root --min-confidence 80
"""

import os
import sys
import argparse
import glob
import json

import numpy as np

DEFAULT_DEPTH_PARM = 1.75
CONFIDENCE_THRESH = 60  # uint8 原始阈值，与 C++ generatePointCloud 一致


# ---------------------------------------------------------------------------
# depthConfig.json
# ---------------------------------------------------------------------------

def parse_depth_scale(depth_dir):
    """从 images_depth/depthConfig.json 读取 depth_parm，计算 scale = depth_parm * 1e-4。

    与 C++ Scene::parse_scale 逻辑一致。
    """
    depth_parm = DEFAULT_DEPTH_PARM
    config_path = os.path.join(depth_dir, "depthConfig.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "depth_parm" in cfg:
                depth_parm = float(cfg["depth_parm"])
                print(f"[INFO] depthConfig.json depth_parm = {depth_parm}")
        except Exception as e:
            print(f"[WARN] 解析 depthConfig.json 失败: {e}，使用默认 {DEFAULT_DEPTH_PARM}")
    else:
        print(f"[INFO] 未找到 depthConfig.json，使用默认 depth_parm = {DEFAULT_DEPTH_PARM}")
    scale = depth_parm * 1e-4
    return scale


# ---------------------------------------------------------------------------
# 等距柱状投影：像素 ↔ 球面方向（与 C++ PanoramicProjector 一致）
# ---------------------------------------------------------------------------

def equirect_pixel_to_ray(row, col, width, height):
    """等距柱状投影像素坐标 → 单位方向向量 (相机坐标系)。

    与 C++ PanoramicProjector::undistortImpl 完全一致：
        lat = (0.5 - row/H) * π
        lon = (0.5 - col/W) * 2π
        x = cos(lat) * sin(lon),  y = sin(lat),  z = cos(lat) * cos(lon)

    Args:
        row, col: 像素坐标，可以是标量或 ndarray。
                  调用时应传 row+0.5, col+0.5 (像素中心)。
        width, height: 全景图尺寸

    Returns:
        directions: (..., 3) 单位向量 [x, y, z]
    """
    row = np.asarray(row, dtype=np.float64)
    col = np.asarray(col, dtype=np.float64)

    lat = (0.5 - row / height) * np.pi
    lon = (0.5 - col / width) * 2.0 * np.pi

    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)

    return np.stack([x, y, z], axis=-1)


def ray_grid(height, width):
    """生成全景图所有像素中心对应的方向向量网格。

    Returns:
        directions: (H, W, 3) 单位向量
    """
    rows = np.arange(height, dtype=np.float64) + 0.5
    cols = np.arange(width, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    return equirect_pixel_to_ray(row_grid, col_grid, width, height)


def world_to_pano_uv(world_points, pose_matrix, width, height):
    """世界坐标 → 全景图像素坐标 (col, row)。

    与 C++ PanoramicProjector::projectImpl 一致：
        lon = atan2(x, z)
        lat = asin(y / len)
        col = (0.5 - lon / 2π) * W
        row = (0.5 - lat / π) * H

    Args:
        world_points: (N, 3) 世界坐标
        pose_matrix: (4, 4) 相机位姿矩阵 (camera-to-world, Twc)
        width, height: 全景图尺寸

    Returns:
        uv: (N, 2) 像素坐标 (col, row)，float
        depth: (N,) 到相机欧氏距离
        valid: (N,) bool，标记是否在图像范围内
    """
    pts = np.asarray(world_points, dtype=np.float64)
    pose = np.asarray(pose_matrix, dtype=np.float64)

    pose_inv = np.linalg.inv(pose)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    homo = np.hstack([pts, ones])
    cam = (pose_inv @ homo.T).T[:, :3]  # (N, 3)

    cx, cy, cz = cam[:, 0], cam[:, 1], cam[:, 2]
    depth = np.sqrt(cx**2 + cy**2 + cz**2)

    safe_depth = np.clip(depth, 1e-8, None)
    lon = np.arctan2(cx, cz)
    lat = np.arcsin(np.clip(cy / safe_depth, -1, 1))

    col = (0.5 - lon / (2.0 * np.pi)) * width
    row = (0.5 - lat / np.pi) * height

    valid = ((col >= 0) & (col < width) &
             (row >= 0) & (row < height) &
             (depth > 1e-6))
    uv = np.stack([col, row], axis=-1)

    return uv, depth, valid


# ---------------------------------------------------------------------------
# 深度图 / 置信度图 加载
# ---------------------------------------------------------------------------

def load_depth(path, scale):
    """加载深度图，返回 float32 (H, W) 单位米。

    与 C++ 一致：depth_meters = raw_uint16 * scale
    """
    import cv2
    raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise FileNotFoundError(f"无法读取深度图: {path}")
    return raw.astype(np.float32) * scale


def load_confidence(path):
    """加载置信度图，返回 uint8 (H, W)。保持原始 0-255 值不归一化。"""
    import cv2
    conf = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if conf is None:
        raise FileNotFoundError(f"无法读取置信度图: {path}")
    if conf.ndim == 3:
        conf = cv2.cvtColor(conf, cv2.COLOR_BGR2GRAY)
    return conf.astype(np.uint8)


# ---------------------------------------------------------------------------
# 深度图 → 相机坐标系点云（不涉及位姿）
# ---------------------------------------------------------------------------

def depth_to_points(depth, confidence=None,
                    min_confidence=CONFIDENCE_THRESH,
                    max_depth=10.0, min_depth=0.001):
    """将单张全景深度图转为相机坐标系点云。

    与 C++ ViewInfo::generatePointCloud 逻辑一致，输出相机局部坐标，不做位姿变换。

    Args:
        depth: (H, W) float32，单位米（已乘 scale）
        confidence: (H, W) uint8，原始 0-255；None 则不做置信度过滤
        min_confidence: uint8 阈值，低于此值的像素丢弃（默认 60）
        max_depth: 深度大于此值的点被丢弃（米，默认 10）
        min_depth: 深度小于此值的点被丢弃（米，默认 0.001）

    Returns:
        points: (N, 3) float32，相机坐标系
        pixel_rc: (N, 2) int32，对应的原始像素坐标 (row, col)
    """
    H, W = depth.shape
    directions = ray_grid(H, W)  # (H, W, 3)

    mask = (depth >= min_depth) & (depth <= max_depth) & np.isfinite(depth)
    if confidence is not None:
        if confidence.shape != depth.shape:
            import cv2
            confidence = cv2.resize(confidence, (W, H),
                                    interpolation=cv2.INTER_NEAREST)
        mask &= confidence >= min_confidence

    rows, cols = np.where(mask)
    d_valid = depth[rows, cols].astype(np.float64)
    dirs_valid = directions[rows, cols]  # (N, 3)

    points = (dirs_valid * d_valid[:, np.newaxis]).astype(np.float32)
    pixel_rc = np.stack([rows, cols], axis=-1).astype(np.int32)

    return points, pixel_rc


def generate_single_pointcloud(depth_path, confidence_path=None,
                               pano_path=None, scale=None, depth_dir=None,
                               min_confidence=CONFIDENCE_THRESH,
                               max_depth=10.0, min_depth=0.001):
    """从单张深度图生成相机坐标系点云。

    Args:
        depth_path: 深度图路径
        confidence_path: 置信度图路径（可选）
        pano_path: 全景 RGB 路径，用于着色（可选）
        scale: 深度缩放系数；None 则从 depth_dir 的 depthConfig.json 读取
        depth_dir: depthConfig.json 所在目录（scale=None 时使用）
        min_confidence, max_depth, min_depth: 过滤阈值

    Returns:
        points: (N, 3) float32，相机坐标系
        colors: (N, 3) uint8 或 None
        pixel_rc: (N, 2) int32 (row, col)
    """
    if scale is None:
        d_dir = depth_dir or os.path.dirname(depth_path)
        scale = parse_depth_scale(d_dir)

    depth = load_depth(depth_path, scale=scale)

    confidence = None
    if confidence_path and os.path.isfile(confidence_path):
        confidence = load_confidence(confidence_path)

    cam_pts, pixel_rc = depth_to_points(
        depth, confidence, min_confidence, max_depth, min_depth)

    colors = None
    if pano_path and os.path.isfile(pano_path) and len(cam_pts) > 0:
        import cv2
        bgr = cv2.imread(pano_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != depth.shape:
                rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            rs, cs = pixel_rc[:, 0], pixel_rc[:, 1]
            colors = rgb[rs, cs].astype(np.uint8)

    return cam_pts, colors, pixel_rc


# ---------------------------------------------------------------------------
# 场景级批量生成（自动扫描 images_depth 目录）
# ---------------------------------------------------------------------------

def generate_scene_pointclouds(depth_dir, output_dir,
                               pano_dir=None,
                               scale=None,
                               min_confidence=CONFIDENCE_THRESH,
                               max_depth=10.0,
                               min_depth=0.001):
    """扫描 depth_dir 中所有 *_depth.png，为每个点位生成相机坐标系点云 PLY。

    不需要 ViewData.txt，不涉及位姿。

    Args:
        depth_dir: 深度图/置信度目录 (images_depth/)
        output_dir: PLY 输出目录
        pano_dir: 全景图目录，用于着色（可选）
        scale: 深度缩放系数；None 则自动从 depthConfig.json 读取
        min_confidence: uint8 阈值（默认 60）
        max_depth, min_depth: 深度过滤阈值（米）

    Returns:
        results: list[dict]，每个点位的生成摘要
    """
    from room_datasets.ply_utils import write_ply

    if scale is None:
        scale = parse_depth_scale(depth_dir)
    print(f"[INFO] depth scale = {scale} (raw_uint16 * {scale} → 米)")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*_depth.png")))
    if not depth_files:
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        depth_files = [f for f in depth_files
                       if not f.endswith("_confidence.png")]
    if not depth_files:
        print(f"[ERROR] {depth_dir} 中未找到深度图")
        return []

    print(f"[INFO] 找到 {len(depth_files)} 个深度图")
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for depth_path in depth_files:
        basename = os.path.basename(depth_path)
        if basename.endswith("_depth.png"):
            name = basename[:-len("_depth.png")]
        else:
            name = os.path.splitext(basename)[0]

        conf_path = os.path.join(depth_dir, f"{name}_confidence.png")

        pano_path = None
        if pano_dir:
            for ext in (".jpg", ".webp", ".png"):
                p = os.path.join(pano_dir, f"{name}{ext}")
                if os.path.isfile(p):
                    pano_path = p
                    break

        cam_pts, colors, _ = generate_single_pointcloud(
            depth_path=depth_path,
            confidence_path=conf_path,
            pano_path=pano_path,
            scale=scale,
            min_confidence=min_confidence,
            max_depth=max_depth,
            min_depth=min_depth,
        )

        if len(cam_pts) == 0:
            print(f"  [EMPTY] {name} — 有效点数为 0")
            continue

        ply_path = os.path.join(output_dir, f"{name}.ply")
        write_ply(ply_path, cam_pts, colors)
        print(f"  [OK  ] {name}  {len(cam_pts)} pts -> {ply_path}")

        results.append({
            "name": name,
            "ply_path": ply_path,
            "num_points": len(cam_pts),
        })

    print(f"[INFO] 共生成 {len(results)} 个点云")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="从全景深度图 + 置信度图生成单点点云 (相机坐标系 PLY)")
    parser.add_argument("input_dir", type=str,
                        help="场景根目录")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="PLY 输出目录（默认: {input_dir}/point_clouds_gen/）")
    parser.add_argument("--depth-dir", type=str, default="",
                        help="深度图/置信度目录（默认: {input_dir}/images_depth/）")
    parser.add_argument("--pano-dir", type=str, default="",
                        help="全景图目录，用于点云着色（默认: {input_dir}/images/）")
    parser.add_argument("--depth-parm", type=float, default=0,
                        help="手动指定 depth_parm（0 = 自动从 depthConfig.json 读取）")
    parser.add_argument("--min-confidence", type=int, default=CONFIDENCE_THRESH,
                        help=f"置信度过滤阈值 uint8（默认 {CONFIDENCE_THRESH}）")
    parser.add_argument("--max-depth", type=float, default=10.0,
                        help="最大深度过滤（米，默认 10）")
    parser.add_argument("--min-depth", type=float, default=0.001,
                        help="最小深度过滤（米，默认 0.001）")
    args = parser.parse_args()

    root = os.path.abspath(args.input_dir)
    if not os.path.isdir(root):
        print(f"[ERROR] 目录不存在: {root}")
        sys.exit(1)

    depth_dir = args.depth_dir or os.path.join(root, "images_depth")
    output_dir = args.output or os.path.join(root, "point_clouds_gen")
    pano_dir = args.pano_dir or os.path.join(root, "images")

    scale = None
    if args.depth_parm > 0:
        scale = args.depth_parm * 1e-4

    print(f"[INFO] 场景目录:   {root}")
    print(f"[INFO] 深度图目录: {depth_dir}")
    print(f"[INFO] 全景图目录: {pano_dir}")
    print(f"[INFO] 输出目录:   {output_dir}")

    results = generate_scene_pointclouds(
        depth_dir=depth_dir,
        output_dir=output_dir,
        pano_dir=pano_dir,
        scale=scale,
        min_confidence=args.min_confidence,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
    )

    if results:
        total = sum(r["num_points"] for r in results)
        print(f"[OK] 共 {len(results)} 个点云，{total} 个点")


if __name__ == "__main__":
    main()
