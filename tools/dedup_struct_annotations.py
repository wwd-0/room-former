#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _line_key_from_ann(ann: Dict[str, Any], round_digits: int) -> Tuple[int, float, float, float, float] | None:
    cat = int(ann.get("category_id", -1))
    if cat not in (16, 17, 18):
        return None

    seg = ann.get("segmentation")
    if not isinstance(seg, list) or len(seg) == 0 or not isinstance(seg[0], list):
        return None
    pts = seg[0]
    if len(pts) < 4:
        return None

    p1 = (float(pts[0]), float(pts[1]))
    p2 = (float(pts[2]), float(pts[3]))
    if p1 > p2:
        p1, p2 = p2, p1

    return (
        cat,
        round(p1[0], round_digits),
        round(p1[1], round_digits),
        round(p2[0], round_digits),
        round(p2[1], round_digits),
    )


def dedup_one_coco(coco: Dict[str, Any], round_digits: int = 1) -> Tuple[Dict[str, Any], Dict[str, int]]:
    anns = coco.get("annotations", [])
    if not isinstance(anns, list):
        raise ValueError("invalid COCO: annotations is not list")

    seen_by_image: Dict[int, set] = defaultdict(set)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    struct_total = 0

    for ann in anns:
        img_id = int(ann.get("image_id", -1))
        key = _line_key_from_ann(ann, round_digits=round_digits)
        if key is None:
            kept.append(ann)
            continue

        struct_total += 1
        if key in seen_by_image[img_id]:
            dropped += 1
            continue
        seen_by_image[img_id].add(key)
        kept.append(ann)

    # 重排 annotation id，避免出现空洞
    for i, ann in enumerate(kept, start=1):
        ann["id"] = i

    new_coco = dict(coco)
    new_coco["annotations"] = kept
    stats = {
        "images": len(coco.get("images", [])),
        "ann_before": len(anns),
        "ann_after": len(kept),
        "struct_total": struct_total,
        "struct_dropped": dropped,
    }
    return new_coco, stats


def _process_file(path: str, round_digits: int, backup: bool) -> None:
    if not os.path.isfile(path):
        print(f"[SKIP] not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    new_coco, stats = dedup_one_coco(coco, round_digits=round_digits)
    if backup:
        bak = path + ".bak"
        with open(bak, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"[BACKUP] {bak}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False)

    print(
        f"[DONE] {path}\n"
        f"  images={stats['images']}  ann_before={stats['ann_before']}  ann_after={stats['ann_after']}\n"
        f"  struct_total={stats['struct_total']}  struct_dropped={stats['struct_dropped']}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="对已生成 COCO 标注去重门/窗/门洞重复线段")
    p.add_argument("--dataset-root", type=str, help="数据集根目录（含 annotations/train.json,val.json）")
    p.add_argument("--ann", type=str, nargs="*", help="直接指定一个或多个 COCO json 文件路径")
    p.add_argument("--round-digits", type=int, default=1, help="线段端点量化精度，默认 1（0.1 像素）")
    p.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份文件")
    args = p.parse_args()

    targets: List[str] = []
    if args.ann:
        targets.extend([os.path.abspath(x) for x in args.ann])
    if args.dataset_root:
        ann_dir = os.path.join(os.path.abspath(args.dataset_root), "annotations")
        targets.extend([os.path.join(ann_dir, "train.json"), os.path.join(ann_dir, "val.json")])

    if not targets:
        p.error("请提供 --dataset-root 或 --ann")

    # 去重 targets，保序
    seen = set()
    uniq_targets = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            uniq_targets.append(t)

    for t in uniq_targets:
        _process_file(t, round_digits=args.round_digits, backup=(not args.no_backup))


if __name__ == "__main__":
    main()

