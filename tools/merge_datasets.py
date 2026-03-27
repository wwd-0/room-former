#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from pathlib import Path

def copy_or_symlink(src, dst, use_symlink):
    if os.path.lexists(dst):
        print(f"[WARN] 目标路径已存在，跳过: {dst}")
        return False
    if use_symlink:
        # 创建相对路经的软链接
        rel = os.path.relpath(os.path.abspath(src), os.path.dirname(os.path.abspath(dst)))
        os.symlink(rel, dst)
    else:
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return True

def merge_coco_json(json_a, json_b, out_json):
    if not os.path.exists(json_a):
        print(f"[ERROR] 找不到标注文件: {json_a}")
        return False
    if not os.path.exists(json_b):
        print(f"[ERROR] 找不到标注文件: {json_b}")
        return False

    with open(json_a, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    with open(json_b, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    # 确定 ID 的偏移量
    max_img_id = max([img['id'] for img in data_a.get('images', [])] + [0])
    max_ann_id = max([ann['id'] for ann in data_a.get('annotations', [])] + [0])

    # 建立 B 中 image_id 的新旧映射关系
    img_id_map = {}
    new_images = []
    for img in data_b.get('images', []):
        old_id = img['id']
        new_id = old_id + max_img_id
        img_id_map[old_id] = new_id
        img['id'] = new_id
        new_images.append(img)

    new_annos = []
    for ann in data_b.get('annotations', []):
        old_img_id = ann['image_id']
        if old_img_id in img_id_map:
            ann['image_id'] = img_id_map[old_img_id]
        
        ann['id'] += max_ann_id
        new_annos.append(ann)

    # 确保选取最全的类别列表（以防其中一个是 16 类，另一个是 20 类）
    cats_a = data_a.get('categories', [])
    cats_b = data_b.get('categories', [])
    merged_categories = cats_a if len(cats_a) >= len(cats_b) else cats_b

    # 合并
    merged_data = {
        "images": data_a.get('images', []) + new_images,
        "annotations": data_a.get('annotations', []) + new_annos,
        "categories": merged_categories,
        "info": {"description": "Merged Dataset", "split": data_a.get('info', {}).get('split', 'unknown')}
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False)
    
    print(f"  [OK] 合并 {os.path.basename(out_json)}:")
    print(f"       数据集A images: {len(data_a.get('images', []))} | 数据集B images: {len(data_b.get('images', []))} -> 合计: {len(merged_data['images'])}")
    print(f"       数据集A annos : {len(data_a.get('annotations', []))} | 数据集B annos : {len(data_b.get('annotations', []))} -> 合计: {len(merged_data['annotations'])}")
    return True

def merge_split_folders(dir_a, dir_b, dir_out, use_symlink):
    os.makedirs(dir_out, exist_ok=True)
    
    count = 0
    # 合并 A
    if os.path.exists(dir_a):
        for scene_name in os.listdir(dir_a):
            src = os.path.join(dir_a, scene_name)
            dst = os.path.join(dir_out, scene_name)
            if copy_or_symlink(src, dst, use_symlink):
                count += 1

    # 合并 B
    if os.path.exists(dir_b):
        for scene_name in os.listdir(dir_b):
            src = os.path.join(dir_b, scene_name)
            dst = os.path.join(dir_out, scene_name)
            if copy_or_symlink(src, dst, use_symlink):
                count += 1
                
    return count

def main():
    parser = argparse.ArgumentParser(description="合并两次 batch_prepare_training_data 跑出来的数据集文件夹结构与 COCO 标注")
    parser.add_argument("--dir1", type=str, required=True, help="第一个数据集根目录 (包含 train, val, annotations)")
    parser.add_argument("--dir2", type=str, required=True, help="第二个数据集根目录")
    parser.add_argument("--out", type=str, required=True, help="合并后的新数据集根目录")
    parser.add_argument("--symlink", action="store_true", help="图软链接物理文件以节省空间（推荐）")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== 开始合并训练集 (Train) ===")
    merge_split_folders(os.path.join(args.dir1, "train"), os.path.join(args.dir2, "train"), str(out_root / "train"), args.symlink)
    merge_coco_json(
        os.path.join(args.dir1, "annotations/train.json"),
        os.path.join(args.dir2, "annotations/train.json"),
        str(out_root / "annotations/train.json")
    )

    print("\n=== 开始合并验证集 (Val) ===")
    merge_split_folders(os.path.join(args.dir1, "val"), os.path.join(args.dir2, "val"), str(out_root / "val"), args.symlink)
    merge_coco_json(
        os.path.join(args.dir1, "annotations/val.json"),
        os.path.join(args.dir2, "annotations/val.json"),
        str(out_root / "annotations/val.json")
    )

    print(f"\n[DONE] 合并完成！输出目录为: {args.out}")
    print("你可以直接用这个新的 dataset_root 来作为 main.py 的参数了。")

if __name__ == "__main__":
    main()
