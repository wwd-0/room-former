#!/usr/bin/env python3
"""
RoomFormer 测试脚本

1. 位姿验证：将各点位点云按位姿合并，输出 merged_poses.ply 供 MeshLab/CloudCompare 目视检查
2. BEV 绘制：加载数据集，绘制房间轮廓并显示/保存

用法:
  python test_pose.py -i /path/to/Datas/ViewData.txt -p /path/to/point_clouds
  python test_pose.py -i /path/to/workspace --draw  # 绘制 BEV 房间轮廓
"""

# macOS 上多个库（numpy/torch/cv2）同时链接 OpenMP 时避免重复初始化错误
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys

# 确保 RoomFormer 在 path 中
ROOMFORMER_ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOMFORMER_ROOT not in sys.path:
    sys.path.insert(0, ROOMFORMER_ROOT)


def find_viewdata(path: str) -> str:
    """在给定路径下查找 ViewData.txt"""
    p = os.path.abspath(path)
    candidates = [
        os.path.join(p, "ViewData.txt"),
        os.path.join(p, "Datas", "ViewData.txt"),
        os.path.join(p, "workspace", "ViewData.txt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def infer_img_dir(viewdata_path: str) -> str:
    """根据 ViewData 路径推断 BasePlan 图像目录"""
    base = os.path.dirname(viewdata_path)
    candidates = [
        os.path.join(base, "BasePlan"),
        os.path.join(os.path.dirname(base), "Datas", "BasePlan"),
        base,
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return base


def _compute_poses_standalone(shoot_spots, base_dir):
    """纯 numpy 位姿计算，使用 pose_utils 中的公共实现。"""
    from datasets.pose_utils import parse_shoot_spots, compute_pose_matrices
    parsed_spots = parse_shoot_spots(shoot_spots, base_dir)
    pose_matrices = compute_pose_matrices(parsed_spots)
    if not pose_matrices:
        print("[ERROR] 未计算出任何位姿矩阵")
        sys.exit(1)
    return parsed_spots, pose_matrices


def _merge_ply_standalone(ply_dir, parsed_spots, pose_matrices, output_path):
    """点云合并，使用 ply_utils 中的公共实现，不依赖 dataLoader/torch。"""
    import numpy as np
    from pathlib import Path
    from datasets.ply_utils import find_ply_files, read_ply, apply_matrix_to_points, write_ply

    ply_files = find_ply_files(ply_dir)
    ply_map = {Path(p).stem: p for p in ply_files}
    if not ply_map:
        print(f"[draw_spots] 未在 {ply_dir} 下找到 .ply 文件")
        return ""
    print(f"[INFO] 目录下共 {len(ply_map)} 个 PLY，开始逐点匹配...")
    print(f"{'spot name':<30} {'ply file':<40} {'pts':>8}")
    all_pts, all_cols = [], []
    for spot, matrix in zip(parsed_spots, pose_matrices):
        name = spot.get("name", "")
        ply_path = ply_map.get(name)
        if ply_path is None:
            print(f"  [SKIP ] name='{name}' -> 未找到对应 PLY")
            continue
        pts, cols = read_ply(ply_path)
        if pts is None or len(pts) == 0:
            print(f"  [EMPTY] name='{name}' -> {Path(ply_path).name} (0 点，跳过)")
            continue
        transformed = apply_matrix_to_points(np.array(matrix, dtype=np.float32), pts)
        all_pts.append(transformed)
        all_cols.append(cols if cols is not None else np.full((len(pts), 3), 128, dtype=np.uint8))
        print(f"  [OK   ] name='{name}' -> {Path(ply_path).name}  ({len(pts)} 点)")
    if not all_pts:
        print("[draw_spots] 未加载到任何有效点云")
        return ""
    merged_pts = np.vstack(all_pts)
    merged_cols = np.vstack(all_cols)
    write_ply(output_path, merged_pts, merged_cols)
    print(f"[draw_spots] 已合并 {len(all_pts)} 个点云，共 {len(merged_pts)} 点 -> {output_path}")
    return output_path


def infer_ply_dir(viewdata_path: str, explicit_ply_dir: str = "") -> str:
    """根据 ViewData 路径推断点云目录"""
    if explicit_ply_dir and os.path.isdir(explicit_ply_dir):
        return explicit_ply_dir
    base = os.path.dirname(viewdata_path)
    # ViewData 可能在 Datas/ 或 workspace/ 下
    candidates = [
        os.path.join(base, "point_clouds"),
        os.path.join(base, "frontend"),
        os.path.join(os.path.dirname(base), "point_clouds"),
        os.path.join(os.path.dirname(base), "frontend"),
        os.path.join(os.path.dirname(base), "workspace_registration", "point_clouds"),
        base,
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return base


def main():
    parser = argparse.ArgumentParser(
        description="位姿验证：将各点位点云按位姿合并，输出 merged_poses.ply 供目视检查"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="ViewData.txt 路径，或包含 ViewData.txt 的目录（如 Datas/、workspace/）",
    )
    parser.add_argument(
        "-p", "--ply-dir",
        type=str,
        default="",
        help="点云目录（默认根据 ViewData 路径自动推断）",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="merged_poses.ply",
        help="合并后的点云输出路径（默认: merged_poses.ply）",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="绘制 BEV 房间轮廓并显示/保存（需指定 img_dir 或 ViewData 同目录下有 BasePlan）",
    )
    parser.add_argument(
        "--draw-output",
        type=str,
        default="",
        help="BEV 绘制结果保存路径（默认不保存，仅显示）",
    )
    parser.add_argument(
        "--use-torch",
        action="store_true",
        help="使用 dataLoader（含 torch）计算位姿；默认用纯 numpy 路径，可避免 macOS segfault",
    )
    args = parser.parse_args()

    viewdata_path = args.input
    if os.path.isdir(viewdata_path):
        viewdata_path = find_viewdata(viewdata_path)
    if not viewdata_path or not os.path.isfile(viewdata_path):
        print(f"[ERROR] 未找到 ViewData.txt: {args.input}")
        sys.exit(1)

    with open(viewdata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    house_data = data.get("HouseData", {})
    shoot_spots = house_data.get("ShootSpots", [])
    if not shoot_spots:
        print("[ERROR] HouseData.ShootSpots 为空")
        sys.exit(1)

    base_dir = os.path.dirname(viewdata_path)
    ply_dir = infer_ply_dir(viewdata_path, args.ply_dir)
    img_dir = infer_img_dir(viewdata_path)

    print(f"[INFO] ViewData: {viewdata_path}")
    print(f"[INFO] 点云目录: {ply_dir}")
    print(f"[INFO] ShootSpots 数量: {len(shoot_spots)}")

    # 默认用纯 numpy 路径，不导入 torch/cv2，避免 macOS segfault
    if args.use_torch:
        from datasets.dataLoader import RoomFormerV1Dataset
        sys.stdout.flush()
        pano_result = RoomFormerV1Dataset.process_panorama_with_depth(
            shoot_spots, base_dir=base_dir, load_images=False
        )
        parsed_spots = pano_result["parsed_spots"]
        pose_matrices = pano_result["pose_matrices"]
        sys.stdout.flush()
        if not pose_matrices:
            print("[ERROR] 未计算出任何位姿矩阵")
            sys.exit(1)
        out_path = RoomFormerV1Dataset.draw_spots(
            ply_dir=ply_dir,
            parsed_spots=parsed_spots,
            pose_matrices=pose_matrices,
            output_path=args.output,
        )
    else:
        parsed_spots, pose_matrices = _compute_poses_standalone(shoot_spots, base_dir)
        out_path = _merge_ply_standalone(ply_dir, parsed_spots, pose_matrices, args.output)


    if out_path:
        print(f"\n[OK] 位姿验证点云已保存: {os.path.abspath(out_path)}")
        print("      可用 MeshLab、CloudCompare 等工具打开查看，检查各点云是否对齐。")
    else:
        print("\n[WARN] 未生成合并点云，请检查点云目录下是否有与 ShootSpots.Name 对应的 .ply 文件")

    # BEV 绘制
    if args.draw:
        import cv2  # 仅在使用 --draw 时导入
        from datasets.dataLoader import RoomFormerV1Dataset
        dataset = RoomFormerV1Dataset(json_path=viewdata_path, img_dir=img_dir)
        if len(dataset) == 0:
            print("[WARN] 数据集为空，跳过 BEV 绘制")
        else:
            data_sample = dataset[0]
            rooms = data_sample["rooms"]
            image = data_sample["image"]
            draw_image = dataset.draw_image(image, rooms)
            if args.draw_output:
                cv2.imwrite(args.draw_output, cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))
                print(f"[OK] BEV 绘制已保存: {os.path.abspath(args.draw_output)}")
            else:
                cv2.imshow("draw_image", cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))
                print("[INFO] 按任意键关闭 BEV 窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if not out_path:
        sys.exit(1)


if __name__ == "__main__":
    main()
