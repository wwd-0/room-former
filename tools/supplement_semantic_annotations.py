#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from typing import Any, Dict, List, Tuple
from pathlib import Path

# ViewData 坐标单位：1 unit = 2 cm = 0.02 m
UNITS_TO_METERS = 0.02

def _poly_bbox_from_pts(pts: np.ndarray, img_w: int, img_h: int, pad: float = 2.0) -> List[float]:
    minx, miny = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    maxx, maxy = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    x0 = max(minx - pad, 0.0)
    y0 = max(miny - pad, 0.0)
    x1 = min(maxx + pad, float(img_w - 1))
    y1 = min(maxy + pad, float(img_h - 1))
    return [x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0)]

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

def _load_gt_from_viewdata(
    viewdata_path: str,
    bev_meta: Dict[str, Any],
    img_w: int,
    img_h: int,
    semantic_rich: bool,
) -> List[Dict[str, Any]]:
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
        world_x = (x_local + room_px) * UNITS_TO_METERS
        world_z = -(z_local + room_pz) * UNITS_TO_METERS
        px = (world_x - origin_x) / mpp
        py = (world_z - origin_z) / mpp
        return px, py

    annos: List[Dict[str, Any]] = []

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

def _default_categories() -> List[Dict[str, Any]]:
    names = [
        "living room", "kitchen", "bedroom", "bathroom", "balcony", "corridor",
        "dining room", "study", "studio", "store room", "garden", "laundry room",
        "office", "basement", "garage", "undefined", "door", "window", "doorway"
    ]
    return [{"supercategory": "room", "id": i, "name": n} for i, n in enumerate(names)]

def main():
    p = argparse.ArgumentParser(description="根据 viewData.txt 重新生成语义标注")
    p.add_argument("--dataset-root", type=str, required=True, help="数据集根目录")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    ann_dir = dataset_root / "annotations"
    
    categories = _default_categories()

    for split in ["train", "val"]:
        json_path = ann_dir / f"{split}.json"
        if not json_path.exists():
            print(f"[WARN] 文件不存在: {json_path}")
            continue

        print(f"[PROCESS] 处理 split: {split}")
        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        new_annotations = []
        ann_id = 1
        
        # 建立 image_id 到 image 信息的映射
        images = coco_data.get("images", [])
        
        for img_info in images:
            file_name = img_info["file_name"]
            img_id = img_info["id"]
            w, h = img_info["width"], img_info["height"]

            # file_name 格式一般为 scene_id/bev_geometric.png
            scene_id = file_name.split("/")[0]
            scene_dir = dataset_root / split / scene_id
            
            viewdata_path = scene_dir / "viewData.txt"
            # 兼容不同的大小写
            if not viewdata_path.exists():
                viewdata_path = scene_dir / "ViewData.txt"
            
            bev_meta_path = scene_dir / "bev_meta.json"

            if not viewdata_path.exists() or not bev_meta_path.exists():
                print(f"[WARN] 场景 {scene_id} 缺少 viewData.txt 或 bev_meta.json，跳过")
                continue

            with open(bev_meta_path, "r", encoding="utf-8") as f:
                bev_meta = json.load(f)

            try:
                scene_anns = _load_gt_from_viewdata(
                    viewdata_path=str(viewdata_path),
                    bev_meta=bev_meta,
                    img_w=w,
                    img_h=h,
                    semantic_rich=True,
                )
            except Exception as e:
                print(f"[FAIL] 场景 {scene_id}: {e}")
                continue

            for a in scene_anns:
                a["id"] = ann_id
                a["image_id"] = img_id
                ann_id += 1
                new_annotations.append(a)

            print(f"[OK] {scene_id}: 生成 {len(scene_anns)} 条标注")

        coco_data["annotations"] = new_annotations
        coco_data["categories"] = categories

        # 备份并保存
        backup_path = json_path.with_suffix(".json.bak")
        os.rename(json_path, backup_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False)
        print(f"[DONE] 已更新 {json_path}，旧文件备份至 {backup_path}")

if __name__ == "__main__":
    main()
