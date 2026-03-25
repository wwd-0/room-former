#!/usr/bin/env python3
"""
验证 GT 标注与 BEV 图像的对齐效果。

功能：
  1. 对单个场景生成 BEV（含填充+缩放）
  2. 使用 _load_gt_from_viewdata 生成标注
  3. 在 BEV 图上绘制房间轮廓、门/窗，类似 test_pose.py 的画法
  4. 保存/显示对比图

用法：
  python verify_gt_alignment.py /path/to/scene_root
  python verify_gt_alignment.py /path/to/scene_root --ply-subdir point_clouds_gen -o verify_output.png
  python verify_gt_alignment.py /path/to/scene_root --show   # 交互显示
"""

import os
import sys
import argparse
import json

import numpy as np

ROOMFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOMFORMER_ROOT not in sys.path:
    sys.path.insert(0, ROOMFORMER_ROOT)

from batch_prepare_training_data import (
    BEV_CANVAS_SIZE, BEV_FINAL_SIZE, UNITS_TO_METERS,
    _load_gt_from_viewdata,
)


# ---------- 颜色方案（BGR）----------
COLOR_ROOM = (0, 0, 255)       # 红：房间轮廓
COLOR_DOOR = (0, 255, 0)       # 绿：门
COLOR_WINDOW = (255, 165, 0)   # 橙：窗
COLOR_CAMERA = (255, 0, 255)   # 紫：相机位置
COLOR_TEXT = (255, 255, 255)   # 白：文字


def draw_annotations_on_bev(bev_img, annos, title=""):
    """在 BEV 图上绘制 COCO 风格标注（房间多边形 + 门窗线段）。"""
    import cv2

    canvas = bev_img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    H, W = canvas.shape[:2]

    room_count = 0
    door_count = 0
    window_count = 0

    for ann in annos:
        cat = ann.get("category_id", 0)
        seg = ann.get("segmentation", [[]])[0]
        if len(seg) < 4:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)

        if cat == 16:  # door
            if len(pts) == 2:
                p1 = tuple(pts[0].astype(int))
                p2 = tuple(pts[1].astype(int))
                cv2.line(canvas, p1, p2, COLOR_DOOR, 2)
                door_count += 1
        elif cat == 17:  # window
            if len(pts) == 2:
                p1 = tuple(pts[0].astype(int))
                p2 = tuple(pts[1].astype(int))
                cv2.line(canvas, p1, p2, COLOR_WINDOW, 2)
                window_count += 1
        else:  # room polygon
            pts_int = pts.astype(np.int32)
            cv2.polylines(canvas, [pts_int], isClosed=True, color=COLOR_ROOM, thickness=2)
            # 在多边形中心标注序号
            cx, cy = pts.mean(axis=0)
            cv2.putText(canvas, f"R{room_count}", (int(cx) - 8, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1,
                        cv2.LINE_AA)
            room_count += 1

    # 图例
    info = f"Rooms: {room_count}, Doors: {door_count}, Windows: {window_count}"
    cv2.putText(canvas, info, (5, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title, (5, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)
    return canvas


def draw_camera_positions_on_bev(canvas, viewdata_path, bev_meta):
    """在 BEV 图上标注相机（ShootSpot）位置。"""
    import cv2

    with open(viewdata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shoot_spots = data.get("HouseData", {}).get("ShootSpots", [])
    if not shoot_spots:
        return canvas

    origin_x = float(bev_meta["origin_x"])
    origin_z = float(bev_meta["origin_z"])
    mpp = float(bev_meta["meters_per_pixel"])

    for spot in shoot_spots:
        pos = spot.get("Position", {})
        px = float(pos.get("x", 0.0))
        pz = float(pos.get("z", 0.0))
        # ShootSpot position → world → BEV pixel
        # 与 compute_pose_matrices 一致：world_x = px*0.02, world_z = -pz*0.02
        world_x = px * UNITS_TO_METERS
        world_z = -pz * UNITS_TO_METERS
        bev_col = (world_x - origin_x) / mpp
        bev_row = (world_z - origin_z) / mpp

        c = int(round(bev_col))
        r = int(round(bev_row))
        H, W = canvas.shape[:2]
        if 0 <= c < W and 0 <= r < H:
            cv2.drawMarker(canvas, (c, r), COLOR_CAMERA,
                           cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
            # 尝试获取点位名称
            thumb = spot.get("ThumbnailUrl", "")
            name = thumb.split("/")[-1].rsplit(".", 1)[0] if thumb else spot.get("Name", "")
            if name:
                cv2.putText(canvas, name, (c + 5, r - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR_CAMERA, 1,
                            cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="验证 GT 标注与 BEV 图像的对齐效果")
    parser.add_argument("scene_root", type=str,
                        help="场景根目录（含 Datas/ViewData.txt + point_clouds_gen/）")
    parser.add_argument("--ply-subdir", type=str, default="point_clouds_gen",
                        help="点云子目录名（默认 point_clouds_gen）")
    parser.add_argument("--mpp", type=float, default=0.02,
                        help="原始 BEV 每像素物理尺寸（米），默认 0.02")
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--semantic", action="store_true",
                        help="生成 semantic-rich 标注（含门窗）")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="保存路径（默认: {scene_root}/verify_gt.png）")
    parser.add_argument("--show", action="store_true",
                        help="交互式显示结果（需 GUI 环境）")
    parser.add_argument("--gen-ply", action="store_true",
                        help="强制重新生成点云")
    args = parser.parse_args()

    import cv2
    from generate_bev import load_and_transform, points_to_bev

    scene_root = os.path.abspath(args.scene_root)
    viewdata = os.path.join(scene_root, "Datas", "ViewData.txt")
    if not os.path.isfile(viewdata):
        print(f"[ERROR] 未找到 {viewdata}")
        sys.exit(1)

    ply_dir = os.path.join(scene_root, args.ply_subdir)

    # 如果需要，先生成点云
    has_ply = (os.path.isdir(ply_dir)
               and any(n.lower().endswith(".ply") for n in os.listdir(ply_dir)))
    if args.gen_ply or not has_ply:
        from generate_pointcloud import generate_scene_pointclouds
        depth_dir = os.path.join(scene_root, "images_depth")
        pano_dir = os.path.join(scene_root, "images")
        os.makedirs(ply_dir, exist_ok=True)
        print("[INFO] 生成点云 ...")
        generate_scene_pointclouds(
            depth_dir=depth_dir,
            output_dir=ply_dir,
            pano_dir=pano_dir if os.path.isdir(pano_dir) else None,
        )

    # 1) 生成 BEV
    print("[INFO] 生成 BEV ...")
    points, scan_ids, scan_names, _ = load_and_transform(viewdata, ply_dir)
    bev_img, meta = points_to_bev(
        points,
        meters_per_pixel=args.mpp,
        output_size=None,
        margin_pixels=args.margin,
        density_log=True,
        scan_ids=scan_ids,
        scan_names=scan_names,
    )
    h_orig, w_orig = bev_img.shape[:2]
    print(f"[INFO] 原始 BEV: {w_orig}×{h_orig} (mpp={args.mpp})")

    # 2) 居中填充 → 1024，缩放 → 256
    canvas = np.zeros((BEV_CANVAS_SIZE, BEV_CANVAS_SIZE, 3), dtype=np.uint8)
    pad_left = (BEV_CANVAS_SIZE - w_orig) // 2
    pad_top = (BEV_CANVAS_SIZE - h_orig) // 2

    dst_x0 = max(pad_left, 0)
    dst_y0 = max(pad_top, 0)
    dst_x1 = min(pad_left + w_orig, BEV_CANVAS_SIZE)
    dst_y1 = min(pad_top + h_orig, BEV_CANVAS_SIZE)
    src_x0 = dst_x0 - pad_left
    src_y0 = dst_y0 - pad_top
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = bev_img[src_y0:src_y1, src_x0:src_x1]

    bev_final = cv2.resize(canvas, (BEV_FINAL_SIZE, BEV_FINAL_SIZE),
                           interpolation=cv2.INTER_AREA)

    ox_raw, oz_raw = meta["origin_world"]
    canvas_ox = ox_raw - pad_left * args.mpp
    canvas_oz = oz_raw - pad_top * args.mpp
    resize_ratio = float(BEV_CANVAS_SIZE) / float(BEV_FINAL_SIZE)
    final_mpp = args.mpp * resize_ratio

    bev_meta = {
        "origin_x": float(canvas_ox),
        "origin_z": float(canvas_oz),
        "meters_per_pixel": float(final_mpp),
    }
    print(f"[INFO] 填充: pad_left={pad_left}, pad_top={pad_top}")
    print(f"[INFO] 最终 BEV: {BEV_FINAL_SIZE}×{BEV_FINAL_SIZE}, mpp={final_mpp:.4f}")
    print(f"[INFO] bev_meta: origin=({canvas_ox:.4f}, {canvas_oz:.4f}), mpp={final_mpp:.4f}")

    # 3) 生成 GT 标注
    anns = _load_gt_from_viewdata(
        viewdata_path=viewdata,
        bev_meta=bev_meta,
        img_w=BEV_FINAL_SIZE,
        img_h=BEV_FINAL_SIZE,
        semantic_rich=args.semantic,
    )
    print(f"[INFO] 标注数: {len(anns)}")

    # 打印每条标注的像素范围
    for i, ann in enumerate(anns):
        seg = ann["segmentation"][0]
        pts = np.array(seg).reshape(-1, 2)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        cat = ann["category_id"]
        cat_name = {0: "room", 15: "room", 16: "door", 17: "window"}.get(cat, f"cat{cat}")
        print(f"  [{i}] {cat_name}: x=[{mn[0]:.1f}, {mx[0]:.1f}] y=[{mn[1]:.1f}, {mx[1]:.1f}]")

    # 4) 绘制
    scene_name = os.path.basename(scene_root)
    result = draw_annotations_on_bev(bev_final, anns, title=scene_name)
    result = draw_camera_positions_on_bev(result, viewdata, bev_meta)

    # 5) 保存 / 显示
    output_path = args.output or os.path.join(scene_root, "verify_gt.png")
    cv2.imwrite(output_path, result)
    print(f"\n[OK] 验证图已保存: {output_path}")

    if args.show:
        # 放大显示
        display = cv2.resize(result, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"GT Alignment - {scene_name}", display)
        print("[INFO] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
