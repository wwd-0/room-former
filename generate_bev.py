#!/usr/bin/env python3
"""
从单点点云 + ShootSpot 位姿生成 3 通道 BEV 底图，并支持 BEV→全景图 反查验证。

输出通道定义：
    R — 点密度        （该像素内点数，log 压缩后归一化到 0-255）
    G — 高度极差      （该像素内 max_y - min_y，归一化到 0-255）
    B — 占用 Mask     （有点 = 255，无点 = 0）

物理约定：1 像素 = 2 cm = 0.02 m

用法：
  # 生成 BEV
  python generate_bev.py /path/to/scene_root

  # 生成 BEV 后进入交互 demo（点击 BEV 像素 → 反查全景图并标注）
  python generate_bev.py /path/to/scene_root --demo

  期望目录结构：
    scene_root/
      ├── Datas/ViewData.txt              # 位姿数据
      ├── point_clouds_gen/*.ply          # 各点位点云（相机坐标系）
      └── images/*.jpg                    # 全景图（demo 模式需要）
"""

import os
import sys
import argparse
import json

import numpy as np

ROOMFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOMFORMER_ROOT not in sys.path:
    sys.path.insert(0, ROOMFORMER_ROOT)

METERS_PER_PIXEL = 0.02  # 1 px = 2 cm


# ---------------------------------------------------------------------------
# 核心：点云 → 3ch BEV 栅格
# ---------------------------------------------------------------------------

def points_to_bev(points, meters_per_pixel=METERS_PER_PIXEL,
                  output_size=None, margin_pixels=10,
                  density_log=True, height_clip_percentile=1.0,
                  scan_ids=None, scan_names=None):
    """将世界坐标点云投影为 3 通道 BEV 图像。

    BEV 采用 XZ 平面（水平面），Y 为高度轴。

    当提供 scan_ids 时：
    - 对密度做多 scan 加权融合，消除重叠区域过曝
    - 构建 scan_coverage (N_scans, H, W) 布尔掩码，
      记录每个像素被哪些 scan 覆盖，可用于反查对应全景图

    Args:
        points: (N, 3+) float32，世界坐标 (x, y, z, ...)
        meters_per_pixel: 每像素对应的物理尺寸（米）
        output_size: 固定输出边长（正方形），None 则自适应
        margin_pixels: 自适应模式下四周留白像素数
        density_log: True 时对密度取 log1p 再归一化（避免热点过曝）
        height_clip_percentile: 高度极差裁切百分位，去掉极端离群值
        scan_ids: (N,) int，每个点所属的 scan 编号（None 则不做融合）
        scan_names: list[str]，scan_id → 点位名称（如 "主卧"），长度 = num_scans

    Returns:
        bev_rgb: uint8 (H, W, 3)  BGR 顺序（方便 cv2 直接写入）
        meta: dict 包含坐标映射、scan 信息等
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        raise ValueError(f"points shape invalid: {pts.shape}")

    wx, wy, wz = pts[:, 0], pts[:, 1], pts[:, 2]

    # ---------- 确定栅格范围 ----------
    x_min, x_max = wx.min(), wx.max()
    z_min, z_max = wz.min(), wz.max()

    if output_size is not None:
        cx = (x_min + x_max) / 2.0
        cz = (z_min + z_max) / 2.0
        half_span = output_size / 2.0 * meters_per_pixel
        x_min_g = cx - half_span
        z_min_g = cz - half_span
        grid_h = grid_w = int(output_size)
    else:
        x_min_g = x_min - margin_pixels * meters_per_pixel
        z_min_g = z_min - margin_pixels * meters_per_pixel
        grid_w = int(np.ceil((x_max - x_min) / meters_per_pixel)) + 2 * margin_pixels
        grid_h = int(np.ceil((z_max - z_min) / meters_per_pixel)) + 2 * margin_pixels

    # ---------- 点→像素坐标 ----------
    pixel_x = ((wx - x_min_g) / meters_per_pixel).astype(np.int32)
    pixel_z = ((wz - z_min_g) / meters_per_pixel).astype(np.int32)

    in_bounds = (pixel_x >= 0) & (pixel_x < grid_w) & (pixel_z >= 0) & (pixel_z < grid_h)
    pixel_x = pixel_x[in_bounds]
    pixel_z = pixel_z[in_bounds]
    wy_valid = wy[in_bounds]

    # ---------- 逐像素统计 ----------
    raw_density = np.zeros((grid_h, grid_w), dtype=np.float64)
    y_min_map = np.full((grid_h, grid_w), np.inf)
    y_max_map = np.full((grid_h, grid_w), -np.inf)

    np.add.at(raw_density, (pixel_z, pixel_x), 1)
    np.minimum.at(y_min_map, (pixel_z, pixel_x), wy_valid)
    np.maximum.at(y_max_map, (pixel_z, pixel_x), wy_valid)

    occupied = raw_density > 0

    # ---------- 多 scan 加权融合 + 覆盖记录 ----------
    scan_coverage = None
    if scan_ids is not None:
        scan_ids_valid = np.asarray(scan_ids, dtype=np.int32)[in_bounds]
        unique_scans = np.unique(scan_ids_valid)
        num_scans = int(scan_ids.max()) + 1 if len(scan_ids) > 0 else 0

        scan_coverage = np.zeros((num_scans, grid_h, grid_w), dtype=np.bool_)
        sum_d_sq = np.zeros((grid_h, grid_w), dtype=np.float64)
        sum_d = np.zeros((grid_h, grid_w), dtype=np.float64)

        for sid in unique_scans:
            mask_s = scan_ids_valid == sid
            d_s = np.zeros((grid_h, grid_w), dtype=np.float64)
            np.add.at(d_s, (pixel_z[mask_s], pixel_x[mask_s]), 1)
            scan_coverage[sid] = d_s > 0
            sum_d_sq += d_s * d_s
            sum_d += d_s

        density = np.where(sum_d > 0, sum_d_sq / sum_d, 0.0)
        fusion_info = {"num_scans": num_scans, "fusion": "weighted_average"}
    else:
        density = raw_density
        fusion_info = {"num_scans": 1, "fusion": "none"}

    # ---------- R: density ----------
    if density_log:
        d = np.log1p(density)
    else:
        d = density.copy()
    d_max = d.max()
    r_ch = (d / d_max * 255).astype(np.uint8) if d_max > 0 else np.zeros_like(d, dtype=np.uint8)

    # ---------- G: height range ----------
    h_range = np.where(occupied, y_max_map - y_min_map, 0.0)
    if height_clip_percentile < 100:
        clip_val = np.percentile(h_range[occupied], 100 - height_clip_percentile) \
            if occupied.any() else 1.0
        h_range = np.clip(h_range, 0, clip_val)
    h_max = h_range.max()
    g_ch = (h_range / h_max * 255).astype(np.uint8) if h_max > 0 else np.zeros_like(h_range, dtype=np.uint8)

    # ---------- B: occupancy mask ----------
    b_ch = (occupied.astype(np.uint8)) * 255

    bev_rgb = np.stack([b_ch, g_ch, r_ch], axis=-1)  # BGR for cv2

    meta = {
        "meters_per_pixel": meters_per_pixel,
        "origin_world": (float(x_min_g), float(z_min_g)),
        "grid_size": (grid_h, grid_w),
        "total_points": int(in_bounds.sum()),
        "density_log": density_log,
        "scan_names": scan_names,
        "scan_coverage": scan_coverage,
        **fusion_info,
    }
    return bev_rgb, meta


def query_pixel_scans(meta, row, col):
    """查询某 BEV 像素由哪些 scan（全景图）贡献。

    Returns:
        list[dict]，每个元素包含 scan_id 和 name（如有）
    """
    coverage = meta.get("scan_coverage")
    names = meta.get("scan_names")
    if coverage is None:
        return []
    h, w = coverage.shape[1], coverage.shape[2]
    if row < 0 or row >= h or col < 0 or col >= w:
        return []
    hit_ids = np.where(coverage[:, row, col])[0]
    results = []
    for sid in hit_ids:
        entry = {"scan_id": int(sid)}
        if names is not None and sid < len(names):
            entry["name"] = names[sid]
        results.append(entry)
    return results


def pixel_to_world(meta, row, col):
    """将 BEV 像素坐标转换为世界坐标 (x, z)。

    Returns:
        (world_x, world_z) 单位米
    """
    mpp = meta["meters_per_pixel"]
    ox, oz = meta["origin_world"]
    return ox + (col + 0.5) * mpp, oz + (row + 0.5) * mpp


# ---------------------------------------------------------------------------
# 点云加载（从 point_clouds_gen/ 读取相机坐标 PLY，应用位姿变换到世界坐标）
# ---------------------------------------------------------------------------

def load_and_transform(viewdata_path, ply_dir):
    """读取 ViewData + 各点位 PLY，变换到世界坐标后合并返回。

    PLY 应为相机坐标系（由 generate_pointcloud.py 生成）。

    Returns:
        merged: (N, 3) 世界坐标点
        scan_ids: (N,) 每个点所属的 scan 编号
        scan_names: list[str]，scan_id → 点位名称
        pose_matrices: list[np.ndarray]，scan_id → 4x4 位姿矩阵 (Twc)
    """
    from room_datasets.pose_utils import parse_shoot_spots, compute_pose_matrices
    from room_datasets.ply_utils import read_ply, apply_matrix_to_points
    from pathlib import Path

    with open(viewdata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shoot_spots = data.get("HouseData", {}).get("ShootSpots", [])
    if not shoot_spots:
        raise RuntimeError("HouseData.ShootSpots 为空")

    base_dir = os.path.dirname(viewdata_path)
    parsed = parse_shoot_spots(shoot_spots, base_dir=base_dir)
    matrices = compute_pose_matrices(parsed)

    ply_map = {}
    if os.path.isdir(ply_dir):
        for f in Path(ply_dir).glob("*.ply"):
            ply_map[f.stem] = str(f)

    all_pts = []
    all_ids = []
    scan_names = []
    scan_matrices = []
    scan_idx = 0
    for spot, mat in zip(parsed, matrices):
        name = spot["name"]
        ply_path = ply_map.get(name)
        if ply_path is None:
            print(f"  [SKIP] {name} — 未找到 PLY")
            continue
        pts, _ = read_ply(ply_path)
        if pts is None or len(pts) == 0:
            continue
        transformed = apply_matrix_to_points(np.array(mat, dtype=np.float32), pts)
        all_pts.append(transformed)
        all_ids.append(np.full(len(pts), scan_idx, dtype=np.int32))
        scan_names.append(name)
        scan_matrices.append(mat)
        scan_idx += 1
        print(f"  [OK  ] {name}  ({len(pts)} pts, scan={scan_idx - 1})")

    if not all_pts:
        raise RuntimeError("未加载到任何有效点云")
    merged = np.vstack(all_pts)
    scan_ids = np.concatenate(all_ids)
    print(f"[INFO] 合并 {len(all_pts)} 个点云（{scan_idx} scans），共 {len(merged)} 点")
    return merged, scan_ids, scan_names, scan_matrices


# ---------------------------------------------------------------------------
# BEV → 全景图反查 demo（交互式）
# ---------------------------------------------------------------------------

def demo_bev_to_pano(bev_img, meta, scan_matrices, pano_dir, scene_root=None):
    """交互式 demo：点击 BEV 图上的像素，反查并标注对应的全景图。

    Args:
        bev_img: (H, W, 3) BGR BEV 图像
        meta: points_to_bev 返回的 meta dict（含 scan_coverage, scan_names）
        scan_matrices: list[np.ndarray]，scan_id → 4x4 Twc 位姿矩阵
        pano_dir: 全景图目录
        scene_root: 场景根目录（仅用于标题显示）
    """
    import cv2
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from generate_pointcloud import world_to_pano_uv

    scan_names = meta.get("scan_names", [])

    bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.imshow(bev_rgb)
    axes.set_title("点击 BEV 图上的任意点 → 反查全景图\n(关闭窗口退出)")
    marker_artists = []

    def on_click(event):
        if event.inaxes != axes:
            return
        bev_col = int(round(event.xdata))
        bev_row = int(round(event.ydata))

        for a in marker_artists:
            a.remove()
        marker_artists.clear()
        dot = axes.plot(bev_col, bev_row, 'r+', markersize=20, markeredgewidth=3)
        marker_artists.extend(dot)
        fig.canvas.draw_idle()

        world_x, world_z = pixel_to_world(meta, bev_row, bev_col)
        print(f"\n{'='*60}")
        print(f"BEV pixel ({bev_row}, {bev_col}) → world ({world_x:.3f}, {world_z:.3f})")

        hits = query_pixel_scans(meta, bev_row, bev_col)
        if not hits:
            print("  该像素无 scan 覆盖")
            return

        print(f"  覆盖 scan: {[h.get('name', h['scan_id']) for h in hits]}")

        pano_results = []
        for h in hits:
            sid = h["scan_id"]
            name = h.get("name", f"scan_{sid}")
            if sid >= len(scan_matrices):
                continue
            pose = scan_matrices[sid]

            world_pt = np.array([[world_x, 0.0, world_z]], dtype=np.float64)
            pano_path = None
            for ext in (".jpg", ".webp", ".png"):
                p = os.path.join(pano_dir, f"{name}{ext}")
                if os.path.isfile(p):
                    pano_path = p
                    break
            if pano_path is None:
                print(f"  [{name}] 全景图不存在，跳过")
                continue

            pano_bgr = cv2.imread(pano_path)
            if pano_bgr is None:
                continue
            pH, pW = pano_bgr.shape[:2]

            uv, depth, valid = world_to_pano_uv(world_pt, pose, pW, pH)
            pcol, prow = uv[0, 0], uv[0, 1]

            print(f"  [{name}] pano ({pW}x{pH}): "
                  f"col={pcol:.1f}, row={prow:.1f}, "
                  f"depth={depth[0]:.2f}m, valid={valid[0]}")

            if valid[0]:
                pano_results.append({
                    "name": name,
                    "pano_bgr": pano_bgr,
                    "col": pcol,
                    "row": prow,
                    "depth": depth[0],
                })

        if not pano_results:
            print("  无有效全景图投影")
            return

        n = len(pano_results)
        fig2, axes2 = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
        for i, pr in enumerate(pano_results):
            ax = axes2[0, i]
            pano_rgb = cv2.cvtColor(pr["pano_bgr"], cv2.COLOR_BGR2RGB)

            r = int(round(pr["row"]))
            c = int(round(pr["col"]))
            radius = 15
            cv2.circle(pano_rgb, (c, r), radius, (255, 0, 0), 3)
            cv2.drawMarker(pano_rgb, (c, r), (255, 0, 0),
                           cv2.MARKER_CROSS, radius * 2, 3)

            ax.imshow(pano_rgb)
            ax.set_title(f"{pr['name']}\npixel=({c},{r})  "
                         f"depth={pr['depth']:.2f}m",
                         fontsize=11)
            ax.axis("off")

        fig2.suptitle(f"BEV ({bev_row},{bev_col}) → "
                      f"world ({world_x:.3f}, {world_z:.3f})",
                      fontsize=13)
        fig2.tight_layout()
        plt.show(block=False)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="从点云 + 位姿生成 3 通道 BEV 底图 (R=密度, G=高度极差, B=Mask)")
    parser.add_argument("input_dir", type=str,
                        help="场景根目录，需包含 Datas/ViewData.txt 和 point_clouds_gen/*.ply")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="输出路径（默认: {input_dir}/bev_geometric.png）")
    parser.add_argument("--ply-dir", type=str, default="",
                        help="点云目录（默认: {input_dir}/point_clouds_gen/）")
    parser.add_argument("--size", type=int, default=None,
                        help="输出正方形边长（像素），默认自适应")
    parser.add_argument("--mpp", type=float, default=METERS_PER_PIXEL,
                        help="每像素物理尺寸（米），默认 0.02 即 2cm")
    parser.add_argument("--margin", type=int, default=10,
                        help="自适应模式下四周留白像素数")
    parser.add_argument("--no-log", action="store_true",
                        help="密度通道不做 log 压缩（线性归一化）")
    parser.add_argument("--no-meta", action="store_true",
                        help="不生成元信息文件")
    parser.add_argument("--demo", action="store_true",
                        help="生成 BEV 后进入交互模式：点击 BEV 像素反查全景图")
    parser.add_argument("--pano-dir", type=str, default="",
                        help="全景图目录（demo 模式用，默认: {input_dir}/images/）")
    args = parser.parse_args()

    root = os.path.abspath(args.input_dir)
    if not os.path.isdir(root):
        print(f"[ERROR] 目录不存在: {root}")
        sys.exit(1)

    viewdata_path = os.path.join(root, "Datas", "ViewData.txt")
    if not os.path.isfile(viewdata_path):
        print(f"[ERROR] 未找到 {viewdata_path}")
        sys.exit(1)

    ply_dir = args.ply_dir or os.path.join(root, "point_clouds_gen")
    if not os.path.isdir(ply_dir):
        print(f"[ERROR] 未找到点云目录 {ply_dir}")
        sys.exit(1)

    output_path = args.output if args.output else os.path.join(root, "bev_geometric.png")
    pano_dir = args.pano_dir or os.path.join(root, "images")

    print(f"[INFO] 场景目录: {root}")
    print(f"[INFO] ViewData:  {viewdata_path}")
    print(f"[INFO] 点云目录:  {ply_dir}")
    print(f"[INFO] 输出:      {output_path}")

    # 1) 加载点云 + 位姿变换
    points, scan_ids, scan_names, scan_matrices = load_and_transform(
        viewdata_path, ply_dir)

    # 2) 生成 BEV
    bev_img, meta = points_to_bev(
        points,
        meters_per_pixel=args.mpp,
        output_size=args.size,
        margin_pixels=args.margin,
        density_log=not args.no_log,
        scan_ids=scan_ids,
        scan_names=scan_names,
    )

    # 3) 保存
    import cv2
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, bev_img)
    print(f"[OK] BEV 已保存: {output_path}  "
          f"({bev_img.shape[1]}x{bev_img.shape[0]}, {meta['total_points']} pts)")

    if not args.no_meta:
        meta_json = {k: v for k, v in meta.items()
                     if k != "scan_coverage"}
        meta_path = os.path.splitext(output_path)[0] + ".json"
        with open(meta_path, "w") as f:
            json.dump(meta_json, f, indent=2, ensure_ascii=False)
        print(f"[OK] 元信息: {meta_path}")

        if meta.get("scan_coverage") is not None:
            npz_path = os.path.splitext(output_path)[0] + "_coverage.npz"
            np.savez_compressed(npz_path, scan_coverage=meta["scan_coverage"])
            print(f"[OK] 覆盖掩码: {npz_path}  "
                  f"(shape {meta['scan_coverage'].shape}, "
                  f"scans: {scan_names})")

    # 4) demo 模式
    if args.demo:
        if not os.path.isdir(pano_dir):
            print(f"[ERROR] 全景图目录不存在: {pano_dir}")
            sys.exit(1)
        print(f"\n[DEMO] 进入交互模式，全景图目录: {pano_dir}")
        print("[DEMO] 在 BEV 图上点击任意位置，将反查对应全景图并标注投影点")
        demo_bev_to_pano(bev_img, meta, scan_matrices, pano_dir,
                         scene_root=root)


if __name__ == "__main__":
    main()
