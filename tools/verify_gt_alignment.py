#!/usr/bin/env python3
"""
验证「已生成的」RoomFormer 训练数据：COCO 标注与 bev_geometric.png 是否对齐。

功能：
  1. 从 annotations/{train|val}.json 读取指定场景的标注
  2. 加载对应 split 下的 bev_geometric.png
  3. 绘制房间轮廓、门/窗/门洞线段
  4. 对每个门/窗/门洞：根据线段中点落在哪个房间内（point-in-polygon，失败则按距房间中心最近）
     从该房间中心向线段两端各画一条细线，标示归属

用法：
  python verify_gt_alignment.py \\
    --dataset-root /path/to/dataset --split train --scene 139263271

  python verify_gt_alignment.py \\
    --dataset-root /path/to/dataset --split val --scene my_id -o /tmp/v.png --show

  # 4 类语义（门/窗/洞为 1/2/3）
  python verify_gt_alignment.py --dataset-root /path/to/dataset --split train --scene x --semantic-classes 4

旧版：从 ViewData 现场生成 BEV+标注（调试用，不再默认）
  python verify_gt_alignment.py --from-viewdata /path/to/scene_root [--semantic] ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

ROOMFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOMFORMER_ROOT not in sys.path:
    sys.path.insert(0, ROOMFORMER_ROOT)


# ---------- 颜色（BGR）----------
COLOR_ROOM = (0, 0, 255)
COLOR_DOOR = (0, 255, 0)
COLOR_WINDOW = (255, 165, 0)
COLOR_OPENING = (255, 0, 255)       # 门洞 / 18
COLOR_ASSOC = (200, 255, 255)       # 房间中心 → 端点的归属连线
COLOR_CENTROID = (0, 255, 255)      # 房间中心点
COLOR_TEXT = (255, 255, 255)


class RoomItem(TypedDict):
    pts: np.ndarray
    cat: int
    centroid: np.ndarray
    idx: int


class OpeningItem(TypedDict):
    p1: np.ndarray
    p2: np.ndarray
    cat: int
    room_idx: int


def _struct_ids(semantic_classes: int) -> frozenset:
    if semantic_classes == 4:
        return frozenset({1, 2, 3})
    if semantic_classes > 0:
        return frozenset({16, 17, 18})
    return frozenset()


def _load_coco_scene(
    dataset_root: str,
    split: str,
    scene_id: str,
) -> Tuple[str, int, int, List[Dict[str, Any]]]:
    root = os.path.abspath(dataset_root)
    ann_path = os.path.join(root, "annotations", f"{split}.json")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(ann_path)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    prefix = f"{scene_id.strip()}/"
    img_rec = None
    for im in images:
        fn = im.get("file_name", "").replace("\\", "/")
        if fn == prefix + "bev_geometric.png" or fn.startswith(prefix):
            img_rec = im
            break
    if img_rec is None:
        raise ValueError(f"在 {ann_path} 中未找到场景 {scene_id!r} 的图像记录")

    iid = int(img_rec["id"])
    w = int(img_rec.get("width", 0))
    h = int(img_rec.get("height", 0))
    anns = [a for a in coco.get("annotations", []) if int(a.get("image_id", -1)) == iid]

    bev_path = os.path.join(root, split, img_rec["file_name"].replace("\\", "/"))
    return bev_path, w, h, anns


def _split_rooms_and_openings(
    anns: List[Dict[str, Any]],
    struct_ids: frozenset,
) -> Tuple[List[RoomItem], List[OpeningItem]]:
    rooms: List[RoomItem] = []
    openings: List[OpeningItem] = []
    ridx = 0

    for ann in anns:
        cat = int(ann.get("category_id", 0))
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg or not isinstance(seg[0], list):
            continue
        flat = seg[0]
        if len(flat) < 4:
            continue

        if cat in struct_ids and len(flat) == 4:
            p1 = np.array([flat[0], flat[1]], dtype=np.float32)
            p2 = np.array([flat[2], flat[3]], dtype=np.float32)
            openings.append({
                "p1": p1, "p2": p2, "cat": cat, "room_idx": -1,
            })
            continue

        if cat in struct_ids:
            continue

        pts = np.array(flat, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        c = pts.mean(axis=0)
        rooms.append({
            "pts": pts,
            "cat": cat,
            "centroid": c,
            "idx": ridx,
        })
        ridx += 1

    return rooms, openings


def _assign_openings_to_rooms(rooms: List[RoomItem], openings: List[OpeningItem]) -> None:
    import cv2

    if not rooms:
        return

    for op in openings:
        p1, p2 = op["p1"], op["p2"]
        mid = (p1 + p2) * 0.5
        mx, my = float(mid[0]), float(mid[1])

        candidates: List[int] = []
        for i, room in enumerate(rooms):
            poly = room["pts"].astype(np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(poly, (mx, my), measureDist=False) >= 0:
                candidates.append(i)

        if len(candidates) == 1:
            op["room_idx"] = candidates[0]
        elif len(candidates) > 1:
            best = min(candidates, key=lambda i: float(np.linalg.norm(rooms[i]["pts"] - mid)))
            op["room_idx"] = best
        else:
            d_cent = [
                (i, float(np.linalg.norm(rooms[i]["centroid"] - mid)))
                for i in range(len(rooms))
            ]
            op["room_idx"] = min(d_cent, key=lambda x: x[1])[0]


def draw_coco_on_bev(
    bev_img: np.ndarray,
    rooms: List[RoomItem],
    openings: List[OpeningItem],
    struct_ids: frozenset,
    title: str = "",
) -> np.ndarray:
    import cv2

    canvas = bev_img.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    H, W = canvas.shape[:2]

    for room in rooms:
        pts_int = room["pts"].astype(np.int32)
        cv2.polylines(canvas, [pts_int], isClosed=True, color=COLOR_ROOM, thickness=2)
        cx, cy = room["centroid"]
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(canvas, c, 4, COLOR_CENTROID, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"R{room['idx']}", (c[0] + 4, c[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)

    n_door = n_win = n_open = 0
    for op in openings:
        cat = op["cat"]
        p1 = (int(round(op["p1"][0])), int(round(op["p1"][1])))
        p2 = (int(round(op["p2"][0])), int(round(op["p2"][1])))
        if cat in (16, 1):
            col = COLOR_DOOR
            n_door += 1
        elif cat in (17, 2):
            col = COLOR_WINDOW
            n_win += 1
        else:
            col = COLOR_OPENING
            n_open += 1
        cv2.line(canvas, p1, p2, col, 3, cv2.LINE_AA)

        ri = op["room_idx"]
        if 0 <= ri < len(rooms):
            c = rooms[ri]["centroid"]
            cc = (int(round(c[0])), int(round(c[1])))
            cv2.line(canvas, cc, p1, COLOR_ASSOC, 1, cv2.LINE_AA)
            cv2.line(canvas, cc, p2, COLOR_ASSOC, 1, cv2.LINE_AA)

    info = (
        f"Rooms={len(rooms)}  Door={n_door}  Win={n_win}  Open={n_open}  "
        f"(Assoc: cyan-ish lines from room centroid to opening endpoints)"
    )
    cv2.putText(canvas, info, (5, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title, (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)
    return canvas


def run_verify_coco_dataset(
    dataset_root: str,
    split: str,
    scene_id: str,
    semantic_classes: int,
    output: str,
    show: bool,
) -> None:
    import cv2

    bev_path, cw, ch, anns = _load_coco_scene(dataset_root, split, scene_id)
    if not os.path.isfile(bev_path):
        raise FileNotFoundError(f"BEV 图像不存在: {bev_path}")

    bev = cv2.imread(bev_path, cv2.IMREAD_UNCHANGED)
    if bev is None:
        raise RuntimeError(f"无法读取图像: {bev_path}")

    if bev.ndim == 2:
        bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    elif bev.shape[2] == 4:
        bev = cv2.cvtColor(bev, cv2.COLOR_BGRA2BGR)

    h, w = bev.shape[:2]
    if cw and ch and (w != cw or h != ch):
        print(f"[WARN] 图像尺寸 {w}x{h} 与 COCO 记录 {cw}x{ch} 不一致，仍以图像为准")

    struct_ids = _struct_ids(semantic_classes)
    rooms, openings = _split_rooms_and_openings(anns, struct_ids)
    _assign_openings_to_rooms(rooms, openings)

    title = f"{split}/{scene_id}"
    result = draw_coco_on_bev(bev, rooms, openings, struct_ids, title=title)

    out_path = output or os.path.join(os.path.dirname(bev_path), "verify_gt_coco.png")
    cv2.imwrite(out_path, result)
    print(f"[OK] {len(anns)} anns -> rooms={len(rooms)} openings={len(openings)}")
    print(f"[OK] saved: {out_path}")

    if show:
        disp = cv2.resize(result, (min(800, w * 3), min(800, h * 3)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("verify_gt_coco", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─── 旧版：ViewData + 现场生成 BEV（可选）────────────────────────────

from batch_prepare_training_data import (  # noqa: E402
    BEV_CANVAS_SIZE,
    BEV_FINAL_SIZE,
    UNITS_TO_METERS,
    _load_gt_from_viewdata,
)

COLOR_CAMERA = (255, 0, 255)


def draw_camera_positions_on_bev(canvas, viewdata_path, bev_meta):
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
        world_x = px * UNITS_TO_METERS
        world_z = -pz * UNITS_TO_METERS
        bev_col = (world_x - origin_x) / mpp
        bev_row = (world_z - origin_z) / mpp
        c, r = int(round(bev_col)), int(round(bev_row))
        H, W = canvas.shape[:2]
        if 0 <= c < W and 0 <= r < H:
            cv2.drawMarker(canvas, (c, r), COLOR_CAMERA, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
            thumb = spot.get("ThumbnailUrl", "")
            name = thumb.split("/")[-1].rsplit(".", 1)[0] if thumb else spot.get("Name", "")
            if name:
                cv2.putText(canvas, name, (c + 5, r - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLOR_CAMERA, 1, cv2.LINE_AA)
    return canvas


def run_from_viewdata(args: argparse.Namespace) -> None:
    import cv2
    from generate_bev import load_and_transform, points_to_bev

    scene_root = os.path.abspath(args.from_viewdata)
    viewdata = os.path.join(scene_root, "Datas", "ViewData.txt")
    if not os.path.isfile(viewdata):
        print(f"[ERROR] 未找到 {viewdata}")
        sys.exit(1)

    ply_dir = os.path.join(scene_root, args.ply_subdir)
    has_ply = os.path.isdir(ply_dir) and any(n.lower().endswith(".ply") for n in os.listdir(ply_dir))
    if args.gen_ply or not has_ply:
        from generate_pointcloud import generate_scene_pointclouds
        depth_dir = os.path.join(scene_root, "images_depth")
        pano_dir = os.path.join(scene_root, "images")
        os.makedirs(ply_dir, exist_ok=True)
        generate_scene_pointclouds(
            depth_dir=depth_dir,
            output_dir=ply_dir,
            pano_dir=pano_dir if os.path.isdir(pano_dir) else None,
        )

    points, scan_ids, scan_names, _ = load_and_transform(viewdata, ply_dir)
    bev_img, meta = points_to_bev(
        points, meters_per_pixel=args.mpp, output_size=None,
        margin_pixels=args.margin, density_log=True,
        scan_ids=scan_ids, scan_names=scan_names,
    )
    h_orig, w_orig = bev_img.shape[:2]
    canvas = np.zeros((BEV_CANVAS_SIZE, BEV_CANVAS_SIZE, 3), dtype=np.uint8)
    pad_left = (BEV_CANVAS_SIZE - w_orig) // 2
    pad_top = (BEV_CANVAS_SIZE - h_orig) // 2
    dst_x0, dst_y0 = max(pad_left, 0), max(pad_top, 0)
    dst_x1 = min(pad_left + w_orig, BEV_CANVAS_SIZE)
    dst_y1 = min(pad_top + h_orig, BEV_CANVAS_SIZE)
    src_x0, src_y0 = dst_x0 - pad_left, dst_y0 - pad_top
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = bev_img[src_y0:src_y1, src_x0:src_x1]
    bev_final = cv2.resize(canvas, (BEV_FINAL_SIZE, BEV_FINAL_SIZE), interpolation=cv2.INTER_AREA)
    ox_raw, oz_raw = meta["origin_world"]
    canvas_ox = ox_raw - pad_left * args.mpp
    canvas_oz = oz_raw - pad_top * args.mpp
    resize_ratio = float(BEV_CANVAS_SIZE) / float(BEV_FINAL_SIZE)
    final_mpp = args.mpp * resize_ratio
    bev_meta = {"origin_x": float(canvas_ox), "origin_z": float(canvas_oz), "meters_per_pixel": float(final_mpp)}

    anns = _load_gt_from_viewdata(
        viewdata_path=viewdata, bev_meta=bev_meta,
        img_w=BEV_FINAL_SIZE, img_h=BEV_FINAL_SIZE,
        semantic_rich=args.semantic,
    )
    struct_ids = _struct_ids(19 if args.semantic else -1)
    rooms, openings = _split_rooms_and_openings(anns, struct_ids)
    _assign_openings_to_rooms(rooms, openings)
    scene_name = os.path.basename(scene_root)
    result = draw_coco_on_bev(bev_final, rooms, openings, struct_ids, title=scene_name)
    result = draw_camera_positions_on_bev(result, viewdata, bev_meta)
    output_path = args.output or os.path.join(scene_root, "verify_gt.png")
    cv2.imwrite(output_path, result)
    print(f"[OK] viewdata 模式已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证 COCO 标注与 BEV 对齐（默认）；可选从 ViewData 生成 BEV")
    parser.add_argument("--dataset-root", type=str, help="数据集根目录（含 train/、annotations/）")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"),
                        help="train / val")
    parser.add_argument("--scene", type=str, default="",
                        help="场景子目录名，如 139263271（对应 train/<scene>/bev_geometric.png）")
    parser.add_argument("--semantic-classes", type=int, default=19,
                        help="19=Structured3D（门/窗/洞 16/17/18）；4=四分类（1/2/3）")
    parser.add_argument("-o", "--output", type=str, default="", help="输出 PNG 路径")
    parser.add_argument("--show", action="store_true", help="弹窗显示")

    parser.add_argument("--from-viewdata", type=str, default="",
                        help="若指定场景根目录，则改为现场生成 BEV+从 ViewData 生成标注（旧逻辑）")
    parser.add_argument("--ply-subdir", type=str, default="point_clouds_gen")
    parser.add_argument("--mpp", type=float, default=0.02)
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument("--semantic", action="store_true",
                        help="与 --from-viewdata 共用：semantic-rich ViewData 标注")
    parser.add_argument("--gen-ply", action="store_true")
    args = parser.parse_args()

    if args.from_viewdata:
        run_from_viewdata(args)
        return

    if not args.dataset_root or not args.scene:
        parser.error("默认模式需要 --dataset-root 与 --scene；或提供 --from-viewdata")

    run_verify_coco_dataset(
        dataset_root=args.dataset_root,
        split=args.split,
        scene_id=args.scene,
        semantic_classes=args.semantic_classes,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
