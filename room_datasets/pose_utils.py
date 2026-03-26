"""
位姿计算工具，不依赖 torch/cv2，供 dataLoader 和 test_pose 共用。
"""

import math
import os
import numpy as np


def euler_position_to_matrix(rotation: tuple, position: tuple) -> np.ndarray:
    """
    由欧拉角 (度) 和位置计算 4x4 变换矩阵 (行主序)。

    旋转顺序: R = Ry * Rx * Rz (YXZ)

    Args:
        rotation: (rx, ry, rz) 欧拉角，单位度
        position: (x, y, z) 平移

    Returns:
        np.ndarray shape (4, 4)，dtype float32
    """
    rx, ry, rz = [math.radians(v) for v in rotation]
    # print("--",rx, ry, rz)
    tx, ty, tz = position
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    r00 = cy * cz + sy * sx * sz
    r01 = -cy * sz + sy * sx * cz
    r02 = sy * cx
    r10 = cx * sz
    r11 = cx * cz
    r12 = -sx
    r20 = -sy * cz + cy * sx * sz
    r21 = sy * sz + cy * sx * cz
    r22 = cy * cx

    return np.array([
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _extract_name(spot: dict) -> str:
    """
    从 ThumbnailUrl 提取点位名称，如 "ThumbnailImages/玄关.jpg" -> "玄关"。
    """
    thumbnail = spot.get("ThumbnailUrl", "")
    if thumbnail:
        return thumbnail.split("/")[-1].rsplit(".", 1)[0]
    return spot.get("Name", "")


def parse_shoot_spots(shoot_spots: list, base_dir: str = "",
                      pano_subdir: str = "", depth_subdir: str = "") -> list:
    """
    解析 HouseData['ShootSpots'] 点位数据，提取每个点位的 ID、名称、位姿及路径。

    名称从 ThumbnailUrl 提取（如 "ThumbnailImages/玄关.jpg" -> "玄关"），
    ThumbnailUrl 缺失时 fallback 到 Name 字段。

    Args:
        shoot_spots: HouseData['ShootSpots'] 列表
        base_dir: 场景根目录
        pano_subdir: 全景图所在子目录（默认 "" 表示与 base_dir 同级）
        depth_subdir: 深度图所在子目录（默认 "" 表示与 base_dir 同级）

    Returns:
        List[dict]，每个元素包含:
            - spot_id, name, position, rotation, panorama_path, depth_path
    """
    pano_dir = os.path.join(base_dir, pano_subdir) if pano_subdir else base_dir
    dep_dir = os.path.join(base_dir, depth_subdir) if depth_subdir else base_dir

    parsed = []
    for spot in shoot_spots or []:
        spot_id = spot.get("ID", "")
        name = _extract_name(spot)
        pos = spot.get("Position", {})
        rot = spot.get("Rotation", {})
        position = (float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0)))
        rotation = (float(rot.get("x", 0)), float(rot.get("y", 0)), float(rot.get("z", 0)))
        parsed.append({
            "spot_id": spot_id,
            "name": name,
            "position": position,
            "rotation": rotation,
            "panorama_path": os.path.join(pano_dir, name + ".jpg"),
            "depth_path": os.path.join(dep_dir, name + "_depth.png"),
        })
    return parsed


UNITS_TO_METERS = 0.02 # ViewData 坐标单位：1 unit = 2 cm = 0.02 m

# 世界坐标系下绕 Y 轴旋转 180°（左乘到变换矩阵上）
_ROT_Y_180 = np.array([
    [-1,  0,  0, 0],
    [ 0,  1,  0, 0],
    [ 0,  0, -1, 0],
    [ 0,  0,  0, 1],
], dtype=np.float32)


def compute_pose_matrices(parsed_spots: list) -> list:
    """
    对 parse_shoot_spots 返回的点位列表批量计算 4x4 变换矩阵。

    位置换算：
        tx =  px * 0.02
        ty =  py * 0.02
        tz = -pz * 0.02   （z 轴取反）

    最终结果再左乘 Y 轴 180° 旋转，将整体转换到目标世界坐标系。

    Args:
        parsed_spots: parse_shoot_spots 返回的列表

    Returns:
        List[np.ndarray]，每个元素为 shape (4, 4) float32 的变换矩阵，与 parsed_spots 一一对应
    """
    matrices = []
    for spot in parsed_spots:
        px, py, pz = spot["position"]
        rx, ry, rz = spot["rotation"]
        tx =  -px * UNITS_TO_METERS
        ty =  py * UNITS_TO_METERS
        tz =  pz * UNITS_TO_METERS
        mat = euler_position_to_matrix((rx, -ry, rz), (tx, ty, tz))
        matrices.append(_ROT_Y_180 @ mat)
    return matrices
