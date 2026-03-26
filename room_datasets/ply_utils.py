"""
PLY 点云读写与变换工具，供 draw_spots 等验证位姿使用。
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def find_ply_files(directory: str) -> List[str]:
    """
    在目录下查找所有 PLY 文件。
    优先查找 point_clouds、frontend 子目录；否则递归当前目录。
    """
    d = Path(directory)
    for sub in ["point_clouds", "frontend"]:
        subdir = d / sub
        if subdir.is_dir():
            files = list(subdir.rglob("*.ply"))
            if files:
                return sorted(str(f) for f in files)
    return sorted(str(f) for f in d.rglob("*.ply"))


def read_ply(ply_path: str, read_colors: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    读取 PLY 文件，返回 points (N,3) 和 colors (N,3) 或 (None, None)。
    """
    ply_to_np = {
        "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
        "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    }
    with open(ply_path, "rb") as f:
        is_ascii = False
        is_big_endian = False
        num_vertices = 0
        vertex_dtype_list = []
        current = None
        while True:
            line = f.readline().decode("latin-1").strip()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                is_ascii = "ascii" in line
                is_big_endian = "binary_big_endian" in line
            elif parts[0] == "element":
                current = parts[1]
                if current == "vertex":
                    num_vertices = int(parts[2])
            elif parts[0] == "property" and current == "vertex":
                t, name = parts[1], parts[-1]
                if t != "list" and t in ply_to_np:
                    dt = ">" + ply_to_np[t] if is_big_endian else "<" + ply_to_np[t]
                    vertex_dtype_list.append((name, dt))
            elif parts[0] == "end_header":
                break
        if num_vertices == 0:
            return np.array([]), None
        if is_ascii:
            data = np.loadtxt(f, max_rows=num_vertices)
            pts = data[:, :3].astype(np.float32)
            cols = data[:, 3:6].astype(np.uint8) if read_colors and data.shape[1] >= 6 else None
            return pts, cols
        if not vertex_dtype_list:
            return np.array([]), None
        dtype = np.dtype(vertex_dtype_list)
        buf = f.read(dtype.itemsize * num_vertices)
        data = np.frombuffer(buf, dtype=dtype, count=num_vertices)
        names = data.dtype.names
        if names is None or "x" not in names:
            return np.array([]), None
        pts = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
        cols = None
        if read_colors:
            r = next((n for n in ["r", "red"] if n in names), None)
            g = next((n for n in ["g", "green"] if n in names), None)
            b = next((n for n in ["b", "blue"] if n in names), None)
            if r and g and b:
                cols = np.column_stack((data[r], data[g], data[b])).astype(np.uint8)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if cols is not None:
            cols = cols[valid]
        return pts, cols


def apply_matrix_to_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """将 4x4 变换矩阵应用到 (N,3) 点集，返回 (N,3)。"""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts, ones])
    out = (matrix @ homo.T).T[:, :3]
    return out


def write_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """将点云写入 PLY 文件（二进制，顶点属性按 vertex 交错存储）。"""
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    has_color = colors is not None and len(colors) == n
    with open(path, "wb") as f:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {n}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        if has_color:
            header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode())
        if has_color:
            cols = np.asarray(colors, dtype=np.uint8)
            for i in range(n):
                f.write(pts[i].tobytes())
                f.write(cols[i].tobytes())
        else:
            pts.tofile(f)
