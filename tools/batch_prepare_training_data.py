#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOMFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOMFORMER_ROOT not in sys.path:
    sys.path.insert(0, ROOMFORMER_ROOT)

# BEV 填充与缩放参数
BEV_CANVAS_SIZE = 1024   # 中间画布尺寸（像素），BEV 居中贴入
BEV_FINAL_SIZE = 256     # 最终输出尺寸（像素），模型期望的输入分辨率


def _copy_or_link(src: str, dst: str, use_symlink: bool) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.lexists(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            return
        os.remove(dst)
    if use_symlink:
        rel = os.path.relpath(os.path.abspath(src), os.path.dirname(os.path.abspath(dst)))
        os.symlink(rel, dst)
    else:
        shutil.copy2(src, dst)


def _copytree_or_link(src: str, dst: str, use_symlink: bool) -> None:
    if not os.path.isdir(src):
        return
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s):
            _copytree_or_link(s, d, use_symlink)
        else:
            _copy_or_link(s, d, use_symlink)


def _remove_source_scene(scene_path: str, scenes_root: str) -> None:
    """删除已处理完的原始场景目录，释放磁盘（仅当路径在 scenes_root 之下时执行）。"""
    root = os.path.abspath(scenes_root)
    path = os.path.abspath(scene_path)
    if path == root or not path.startswith(root + os.sep):
        print(f"[WARN] 拒绝删除源目录（不在 --scenes-root 之下）: {scene_path}")
        return
    if not os.path.isdir(path):
        return
    shutil.rmtree(path)


def _discover_scenes(scenes_root: str) -> List[str]:
    root = Path(scenes_root)
    if not root.is_dir():
        raise FileNotFoundError(scenes_root)
    scenes = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        vd = p / "Datas" / "ViewData.txt"
        if vd.is_file():
            scenes.append(str(p))
    return scenes


def _load_gt_annotations(gt_path: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """从 floorplan_gt.json 解析 COCO 风格 annotation 列表。"""
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise RuntimeError("需要 shapely：pip install shapely")

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("annotations", data if isinstance(data, list) else [])
    out = []
    for i, obj in enumerate(raw):
        cat = int(obj.get("category_id", 0))
        seg = obj.get("segmentation")
        if not seg or not seg[0]:
            continue
        flat = list(seg[0])
        if len(flat) < 6:
            continue
        pts = np.array(flat, dtype=np.float64).reshape(-1, 2)
        poly = Polygon(pts)
        if not poly.is_valid or poly.area < 1e-6:
            continue
        minx, miny, maxx, maxy = poly.bounds
        pad = 2.0
        x0 = max(minx - pad, 0.0)
        y0 = max(miny - pad, 0.0)
        x1 = min(maxx + pad, float(img_w - 1))
        y1 = min(maxy + pad, float(img_h - 1))
        bb = [x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0)]
        out.append({
            "segmentation": [flat],
            "area": float(poly.area),
            "iscrowd": 0,
            "bbox": bb,
            "category_id": cat,
        })
    return out


def _poly_bbox_from_pts(pts: np.ndarray, img_w: int, img_h: int, pad: float = 2.0) -> List[float]:
    minx, miny = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    maxx, maxy = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    x0 = max(minx - pad, 0.0)
    y0 = max(miny - pad, 0.0)
    x1 = min(maxx + pad, float(img_w - 1))
    y1 = min(maxy + pad, float(img_h - 1))
    return [x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0)]


UNITS_TO_METERS = 0.02  # ViewData 坐标单位：1 unit = 2 cm = 0.02 m


def _load_gt_from_viewdata(
    viewdata_path: str,
    bev_meta: Dict[str, Any],
    img_w: int,
    img_h: int,
    semantic_rich: bool,
) -> List[Dict[str, Any]]:
    """从 ViewData.txt 自动提取房间/门窗标注。

    使用 bev_meta 中的 origin_x/z 和 meters_per_pixel 将 ViewData 坐标
    正确映射到最终 BEV 像素空间（与 generate_bev / compute_pose_matrices 一致）。
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise RuntimeError("需要 shapely：pip install shapely")

    with open(viewdata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    floors = data.get("HouseData", {}).get("Floors", [])
    if not floors:
        return []
    floor = floors[0]
    rooms = floor.get("Rooms", [])

    # 从 bev_meta 中读取坐标映射参数
    origin_x = float(bev_meta["origin_x"])
    origin_z = float(bev_meta["origin_z"])
    mpp = float(bev_meta["meters_per_pixel"])

    def _to_bev_px(x_local: float, z_local: float,
                   room_px: float, room_pz: float) -> Tuple[float, float]:
        # 1) ViewData 局部坐标 → 世界坐标（米）
        #    与 compute_pose_matrices 一致：x 同向，z 取反
        world_x = (x_local + room_px) * UNITS_TO_METERS
        world_z = -(z_local + room_pz) * UNITS_TO_METERS
        # 2) 世界坐标 → BEV 像素
        px = (world_x - origin_x) / mpp
        py = (world_z - origin_z) / mpp
        return px, py

    annos: List[Dict[str, Any]] = []

    def _map_room_name_to_id(name: str) -> int:
        if not name: return 15
        if '客' in name and '厅' in name: return 0
        if '厨' in name: return 1
        if '卧' in name or '儿童房' in name or '老人房' in name: return 2
        if '卫' in name or '厕' in name or '洗手' in name or '浴' in name: return 3
        if '阳台' in name or '露台' in name: return 4
        if '走廊' in name or '过道' in name or '玄关' in name: return 5
        if '餐' in name: return 6
        if '书房' in name: return 7
        if '工作室' in name: return 8
        if '衣帽' in name or '储' in name or '杂物' in name: return 9
        if '花园' in name or '庭院' in name: return 10
        if '洗衣' in name: return 11
        if '办公' in name: return 12
        if '地下' in name: return 13
        if '车' in name: return 14
        return 15

    for room in rooms:
        info = room.get("Info", {})
        room_name = info.get("Name", "") or room.get("Name", "") or room.get("RoomName", "")
        pos = info.get("Position", {})
        room_px = float(pos.get("x", 0.0))
        room_pz = float(pos.get("z", 0.0))

        verts: List[List[float]] = []
        for wall in room.get("Walls", []):
            try:
                s = wall["Start"]["Up"]["Position"]
                e = wall["End"]["Up"]["Position"]
                sxp, syp = _to_bev_px(float(s["x"]), float(s["z"]), room_px, room_pz)
                exp, eyp = _to_bev_px(float(e["x"]), float(e["z"]), room_px, room_pz)
                verts.append([sxp, syp])
                verts.append([exp, eyp])
            except Exception:
                continue

        if len(verts) >= 3:
            pts = np.array(verts, dtype=np.float64)
            # 去重，保留顺序
            uniq = []
            for p in pts:
                if not uniq or np.linalg.norm(p - uniq[-1]) > 1e-3:
                    uniq.append(p)
            pts = np.array(uniq, dtype=np.float64)
            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area >= 1.0:
                    x, y = poly.exterior.coords.xy
                    ring = np.column_stack([x[:-1], y[:-1]])
                    flat = ring.reshape(-1).tolist()
                    
                    cat_id = 0
                    if semantic_rich:
                        cat_id = _map_room_name_to_id(room_name)

                    annos.append({
                        "segmentation": [flat],
                        "area": float(poly.area),
                        "iscrowd": 0,
                        "bbox": _poly_bbox_from_pts(ring, img_w, img_h),
                        "category_id": cat_id,
                    })

        if semantic_rich:
            for key, base_cat_id in [("Doors", 16), ("Windows", 17)]:
                for obj in room.get(key, []):
                    try:
                        # 区分真门和门洞（OpenArea）
                        current_cat_id = base_cat_id
                        if key == "Doors" and obj.get("Type", "Door") == "OpenArea":
                            current_cat_id = 18

                        s = obj["Start"]["Up"]["Position"]
                        e = obj["End"]["Up"]["Position"]
                        sxp, syp = _to_bev_px(float(s["x"]), float(s["z"]), room_px, room_pz)
                        exp, eyp = _to_bev_px(float(e["x"]), float(e["z"]), room_px, room_pz)
                        seg = [sxp, syp, exp, eyp]
                        line_pts = np.array([[sxp, syp], [exp, eyp]], dtype=np.float64)
                        annos.append({
                            "segmentation": [seg],
                            "area": 1.0,
                            "iscrowd": 0,
                            "bbox": _poly_bbox_from_pts(line_pts, img_w, img_h, pad=1.0),
                            "category_id": current_cat_id,
                        })
                    except Exception:
                        continue

    return annos


def _default_categories(semantic_rich: bool) -> List[Dict[str, Any]]:
    if not semantic_rich:
        return [{"supercategory": "room", "id": 0, "name": "room"}]
    # Structured3D 语义（与 data_preprocess/stru3d 一致）
    names = [
        "living room", "kitchen", "bedroom", "bathroom", "balcony", "corridor",
        "dining room", "study", "studio", "store room", "garden", "laundry room",
        "office", "basement", "garage", "undefined", "door", "window", "doorway"
    ]
    return [{"supercategory": "room", "id": i, "name": n} for i, n in enumerate(names)]


def _build_coco_split(
    entries: List[Tuple[str, str, int, int, List[Dict]]],
    categories: List[Dict],
    split_name: str,
) -> Dict[str, Any]:
    images = []
    annotations = []
    ann_id = 1
    for img_id, (rel_name, scene_id, w, h, anns) in enumerate(entries, start=1):
        images.append({
            "file_name": rel_name.replace("\\", "/"),
            "id": img_id,
            "width": w,
            "height": h,
        })
        for a in anns:
            ad = dict(a)
            ad["image_id"] = img_id
            ad["id"] = ann_id
            ann_id += 1
            annotations.append(ad)
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {"description": f"RoomFormer {split_name}", "split": split_name},
    }


def _merge_coco_append(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """按 file_name 去重追加 COCO，保留 existing 的历史样本。"""
    ex_images = list(existing.get("images", []))
    ex_anns = list(existing.get("annotations", []))
    categories = existing.get("categories") or new_data.get("categories", [])
    info = existing.get("info") or new_data.get("info", {})

    keep_file_names = {img.get("file_name", "") for img in ex_images}
    old_to_new_id: Dict[int, int] = {}
    next_img_id = max((int(img.get("id", 0)) for img in ex_images), default=0) + 1
    for img in new_data.get("images", []):
        fn = img.get("file_name", "")
        if fn in keep_file_names:
            continue
        old_id = int(img["id"])
        new_id = next_img_id
        next_img_id += 1
        old_to_new_id[old_id] = new_id
        ni = dict(img)
        ni["id"] = new_id
        ex_images.append(ni)
        keep_file_names.add(fn)

    next_ann_id = max((int(a.get("id", 0)) for a in ex_anns), default=0) + 1
    for ann in new_data.get("annotations", []):
        old_img_id = int(ann.get("image_id", -1))
        if old_img_id not in old_to_new_id:
            continue
        na = dict(ann)
        na["id"] = next_ann_id
        next_ann_id += 1
        na["image_id"] = old_to_new_id[old_img_id]
        ex_anns.append(na)

    return {
        "images": ex_images,
        "annotations": ex_anns,
        "categories": categories,
        "info": info,
    }


def process_one_scene(
    scene_root: str,
    scene_out: str,
    ply_subdir: str,
    bev_size: Optional[int],
    mpp: float,
    margin: int,
    density_log: bool,
    save_bev_extra: bool,
    gen_ply: bool,
    symlink: bool,
) -> Tuple[int, int, Dict[str, Any]]:
    """生成 BEV + bev_meta，并拷贝全景/深度/viewData。返回 (img_w, img_h, meta)。"""
    from generate_bev import load_and_transform, points_to_bev

    scene_root = os.path.abspath(scene_root)
    viewdata = os.path.join(scene_root, "Datas", "ViewData.txt")
    if not os.path.isfile(viewdata):
        raise FileNotFoundError(viewdata)

    ply_dir = os.path.join(scene_root, ply_subdir)
    has_existing_ply = (
        os.path.isdir(ply_dir)
        and any(name.lower().endswith(".ply") for name in os.listdir(ply_dir))
    )
    need_generate_ply = gen_ply or (not has_existing_ply)
    if need_generate_ply:
        from generate_pointcloud import generate_scene_pointclouds

        depth_dir = os.path.join(scene_root, "images_depth")
        pano_dir = os.path.join(scene_root, "images")
        os.makedirs(ply_dir, exist_ok=True)
        if not has_existing_ply:
            print(f"[INFO] {os.path.basename(scene_root)}: 未发现 {ply_subdir}/*.ply，自动生成点云")
        else:
            print(f"[INFO] {os.path.basename(scene_root)}: 按 --gen-ply 重新生成点云")
        generate_scene_pointclouds(
            depth_dir=depth_dir,
            output_dir=ply_dir,
            pano_dir=pano_dir if os.path.isdir(pano_dir) else None,
        )

    points, scan_ids, scan_names, _mats = load_and_transform(viewdata, ply_dir)
    bev_img, meta = points_to_bev(
        points,
        meters_per_pixel=mpp,
        output_size=bev_size,
        margin_pixels=margin,
        density_log=density_log,
        scan_ids=scan_ids,
        scan_names=scan_names,
    )

    import cv2

    # ------------------------------------------------------------------
    # 居中填充到 BEV_CANVAS_SIZE × BEV_CANVAS_SIZE，再缩放到 BEV_FINAL_SIZE × BEV_FINAL_SIZE
    # ------------------------------------------------------------------
    h_orig, w_orig = bev_img.shape[:2]
    canvas = np.zeros((BEV_CANVAS_SIZE, BEV_CANVAS_SIZE, 3), dtype=np.uint8)

    # 计算居中偏移（可能为负数，表示 BEV 超出画布）
    pad_left = (BEV_CANVAS_SIZE - w_orig) // 2
    pad_top = (BEV_CANVAS_SIZE - h_orig) // 2

    # 画布上的目标区域
    dst_x0 = max(pad_left, 0)
    dst_y0 = max(pad_top, 0)
    dst_x1 = min(pad_left + w_orig, BEV_CANVAS_SIZE)
    dst_y1 = min(pad_top + h_orig, BEV_CANVAS_SIZE)
    # 原图上的对应区域
    src_x0 = dst_x0 - pad_left
    src_y0 = dst_y0 - pad_top
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = bev_img[src_y0:src_y1, src_x0:src_x1]

    if h_orig > BEV_CANVAS_SIZE or w_orig > BEV_CANVAS_SIZE:
        print(f"[WARN] {os.path.basename(scene_root)}: BEV ({w_orig}×{h_orig}) "
              f"超过画布 {BEV_CANVAS_SIZE}，已裁剪边缘")

    # 缩放到最终尺寸
    bev_final = cv2.resize(canvas, (BEV_FINAL_SIZE, BEV_FINAL_SIZE),
                           interpolation=cv2.INTER_AREA)

    # ---------- 更新物理参数 ----------
    ox_raw, oz_raw = meta["origin_world"]
    # 1024 画布左上角对应的世界坐标 = 原始原点 − 填充像素 × 原始 mpp
    canvas_ox = ox_raw - pad_left * mpp
    canvas_oz = oz_raw - pad_top * mpp
    # 缩放后每像素对应的物理尺寸 = 原始 mpp × (画布 / 最终)
    resize_ratio = float(BEV_CANVAS_SIZE) / float(BEV_FINAL_SIZE)  # 4.0
    final_mpp = mpp * resize_ratio  # 0.02 * 4 = 0.08 m/px

    os.makedirs(scene_out, exist_ok=True)
    bev_path = os.path.join(scene_out, "bev_geometric.png")
    cv2.imwrite(bev_path, bev_final)
    w, h = BEV_FINAL_SIZE, BEV_FINAL_SIZE

    bev_meta = {
        "origin_x": float(canvas_ox),
        "origin_z": float(canvas_oz),
        "meters_per_pixel": float(final_mpp),
        "world_y": 0.0,
        "grid_height": BEV_FINAL_SIZE,
        "grid_width": BEV_FINAL_SIZE,
        "original_mpp": float(mpp),
        "canvas_size": BEV_CANVAS_SIZE,
        "final_size": BEV_FINAL_SIZE,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "original_h": h_orig,
        "original_w": w_orig,
    }
    with open(os.path.join(scene_out, "bev_meta.json"), "w", encoding="utf-8") as f:
        json.dump(bev_meta, f, indent=2, ensure_ascii=False)

    if save_bev_extra:
        meta_json = {k: v for k, v in meta.items() if k != "scan_coverage"}
        with open(os.path.join(scene_out, "bev_geometric.json"), "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2, ensure_ascii=False)
        if meta.get("scan_coverage") is not None:
            npz_path = os.path.join(scene_out, "bev_geometric_coverage.npz")
            np.savez_compressed(npz_path, scan_coverage=meta["scan_coverage"])

    # viewData：训练时 poly_data 在 scene 根下找 viewData.txt
    src_vd = viewdata
    dst_vd = os.path.join(scene_out, "viewData.txt")
    _copy_or_link(src_vd, dst_vd, symlink)

    img_src = os.path.join(scene_root, "images")
    dep_src = os.path.join(scene_root, "images_depth")
    if os.path.isdir(img_src):
        _copytree_or_link(img_src, os.path.join(scene_out, "panoramas"), symlink)
    if os.path.isdir(dep_src):
        _copytree_or_link(dep_src, os.path.join(scene_out, "depths"), symlink)

    return w, h, bev_meta


def main():
    p = argparse.ArgumentParser(description="批量生成 RoomFormer 训练数据目录与 COCO JSON")
    p.add_argument("--scenes-root", type=str, required=True,
                   help="父目录：其下每个子目录为一套场景（含 Datas/ViewData.txt）")
    p.add_argument("--out", type=str, required=True, help="输出 dataset 根目录")
    p.add_argument("--val-ratio", type=float, default=0.1, help="验证集场景比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ply-subdir", type=str, default="point_clouds_gen",
                   help="场景内点云子目录名")
    p.add_argument("--bev-size", type=int, default=None,
                   help="BEV 正方形边长像素，默认自适应")
    p.add_argument("--mpp", type=float, default=0.02, help="米/像素，默认 2cm")
    p.add_argument("--margin", type=int, default=10)
    p.add_argument("--no-log", action="store_true", help="BEV 密度不做 log")
    p.add_argument("--save-bev-extra", action="store_true",
                   help="额外保存 bev_geometric.json 与 coverage npz")
    p.add_argument("--gen-ply", action="store_true",
                   help="强制从 images_depth 重新生成点云；默认仅在缺少 ply 时自动生成")
    p.add_argument("--symlink", action="store_true", help="全景/深度用软链")
    p.add_argument("--delete-source", action="store_true",
                   help="每场成功导出后删除该场景在 --scenes-root 下的原始目录，节省磁盘；与 --symlink 互斥")
    p.add_argument("--no-semantic-categories", dest="semantic_categories", action="store_false",
                   help="关闭语义类别（默认开启 19 类 Structured3D 语义表）")
    p.set_defaults(semantic_categories=True)
    p.add_argument("--gt-name", type=str, default="floorplan_gt.json",
                   help="各场景根目录下 GT 文件名；不存在时自动从 ViewData.txt 转换生成")
    p.add_argument("--allow-empty-gt", action="store_true",
                   help="允许无 floorplan_gt.json 的场景仍写入图像（标注为空）")
    p.add_argument("--append", action="store_true",
                   help="增量模式：若 annotations 已存在则追加新场景，不覆盖旧标注")
    args = p.parse_args()

    if args.delete_source and args.symlink:
        print("[ERROR] --delete-source 与 --symlink 不能同时使用（删除源会破坏软链）")
        sys.exit(1)

    scenes = _discover_scenes(args.scenes_root)
    if not scenes:
        print(f"[ERROR] 在 {args.scenes_root} 下未发现含 Datas/ViewData.txt 的子目录")
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(scenes)
    n_val = max(1, int(len(scenes) * args.val_ratio)) if len(scenes) > 1 else 0
    if len(scenes) == 1:
        n_val = 0
    val_set = set(scenes[:n_val])
    train_scenes = [s for s in scenes if s not in val_set]
    val_scenes = [s for s in scenes if s in val_set]

    out_root = os.path.abspath(args.out)
    train_dir = os.path.join(out_root, "train")
    val_dir = os.path.join(out_root, "val")
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    categories = _default_categories(args.semantic_categories)

    train_entries: List[Tuple[str, str, int, int, List]] = []
    val_entries: List[Tuple[str, str, int, int, List]] = []
    skipped = []

    def handle_split(scene_path: str, split_dir: str, bucket: List):
        sid = os.path.basename(os.path.normpath(scene_path))
        gt_path = os.path.join(scene_path, args.gt_name)
        scene_out = os.path.join(split_dir, sid)
        try:
            w, h, bev_meta = process_one_scene(
                scene_path,
                scene_out,
                ply_subdir=args.ply_subdir,
                bev_size=args.bev_size,
                mpp=args.mpp,
                margin=args.margin,
                density_log=not args.no_log,
                save_bev_extra=args.save_bev_extra,
                gen_ply=args.gen_ply,
                symlink=args.symlink,
            )
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
            skipped.append((sid, str(e)))
            return

        # file_name 相对 img_folder（train/ 或 val/），与 datasets/poly_data 一致
        rel = f"{sid}/bev_geometric.png"
        anns: List[Dict] = []
        if os.path.isfile(gt_path):
            anns = _load_gt_annotations(gt_path, w, h)
        else:
            viewdata = os.path.join(scene_path, "Datas", "ViewData.txt")
            anns = _load_gt_from_viewdata(
                viewdata_path=viewdata,
                bev_meta=bev_meta,
                img_w=w,
                img_h=h,
                semantic_rich=args.semantic_categories,
            )
            if anns:
                print(f"[INFO] {sid}: 无 {args.gt_name}，已从 ViewData.txt 自动生成 {len(anns)} 条标注")
            elif not args.allow_empty_gt:
                print(f"[SKIP] {sid}: 无 {args.gt_name} 且 ViewData 未生成有效标注（加 --allow-empty-gt 可仅导出图像）")
                shutil.rmtree(scene_out, ignore_errors=True)
                skipped.append((sid, "no_gt"))
                return

        bucket.append((rel, sid, w, h, anns))
        print(f"[OK] {sid} -> {scene_out}  ({w}x{h}, {len(anns)} annos)")

        if args.delete_source:
            _remove_source_scene(scene_path, args.scenes_root)
            print(f"[INFO] {sid}: 已删除源场景目录以释放空间")

    for s in train_scenes:
        handle_split(s, train_dir, train_entries)
    for s in val_scenes:
        handle_split(s, val_dir, val_entries)

    train_coco = _build_coco_split(train_entries, categories, "train")
    val_coco = _build_coco_split(val_entries, categories, "val")

    train_json_path = os.path.join(ann_dir, "train.json")
    val_json_path = os.path.join(ann_dir, "val.json")
    if args.append:
        if os.path.isfile(train_json_path):
            with open(train_json_path, "r", encoding="utf-8") as f:
                train_coco = _merge_coco_append(json.load(f), train_coco)
        if os.path.isfile(val_json_path):
            with open(val_json_path, "r", encoding="utf-8") as f:
                val_coco = _merge_coco_append(json.load(f), val_coco)

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)


    readme = os.path.join(out_root, "README_dataset.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "RoomFormer 数据根目录\n"
            f"训练场景数: {len(train_entries)}, 验证: {len(val_entries)}\n"
            "训练命令示例:\n"
            f"  python main.py --dataset_root {out_root} ... \\\n"
            "    --use_pano --pano_dir <同 dataset_root> \\\n"
            "    --pano_backproject_bias   # 若各场景有 bev_meta.json\n"
        )

    print(f"\n[DONE] 输出: {out_root}")
    print(f"  annotations/train.json  images={len(train_coco['images'])}  "
          f"annos={len(train_coco['annotations'])}")
    print(f"  annotations/val.json    images={len(val_coco['images'])}  "
          f"annos={len(val_coco['annotations'])}")

    if skipped:
        print(f"  失败/跳过: {len(skipped)} 条")


if __name__ == "__main__":
    main()
