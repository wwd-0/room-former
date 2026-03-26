import json
import cv2
import os
import numpy as np
import torch

class RoomFormerV1Dataset():
    TARGET_SIZE = 1024

    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)  # 标注数据
        self.img_dir = img_dir
        self.transform = transform
        self.pixel_to_meter = 0.02  # 1像素=2cm

    def __len__(self):
        return len(self.data)

    # -------------------------------------------------------------------------
    # 图像转张量 / 多模态输入接口（供模型训练使用）
    # -------------------------------------------------------------------------

    @staticmethod
    def bev_image_to_tensor(image: np.ndarray) -> torch.Tensor:
        """
        将 BEV 图像转为模型输入张量。

        - 像素值归一化到 [0, 1]
        - 通道顺序调整为 [C, H, W]

        Args:
            image: BGR 图像，shape (H, W, 3)，dtype uint8 或 float

        Returns:
            torch.Tensor, shape (3, H, W), dtype float32, 范围 [0, 1]
        """
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(arr).float()

    @staticmethod
    def process_panorama_with_depth(
        panorama_path: str,
        depth_path: str,
        **kwargs,
    ) -> torch.Tensor:
        """
        全景图与深度图合并处理接口，供模型训练使用。

        可将 RGB 全景图与深度图合并为 4 通道 [R, G, B, D] 输入模型。

        Args:
            panorama_path: 全景图文件路径
            depth_path: 深度图文件路径
            **kwargs: 后续扩展参数

        Returns:
            处理后的张量，shape 如 (4, H, W)，具体格式待实现。
        """
        raise NotImplementedError("process_panorama_with_depth 待实现")

    def draw_image(self, image, rooms):
        if image is None:
            print("[draw_image] image 为 None，跳过绘制")
            return image

        # 图像已为 1024×1024，rooms 中坐标为归一化 [0,1]，绘制时需反归一化
        canvas = image.copy()
        H, W = canvas.shape[:2]

        def _denorm(pts):
            """归一化坐标 [0,1] 转为像素坐标。"""
            if not pts:
                return np.array([], dtype=np.float32).reshape(0, 2)
            arr = np.array(pts, dtype=np.float32)
            arr[:, 0] *= W
            arr[:, 1] *= H
            return arr

        def _clip_pts(pts):
            """过滤越界点，返回合法的 int32 数组；不足 2 点则返回 None。"""
            if not pts:
                return None
            arr = _denorm(pts)
            in_bounds = (
                (arr[:, 0] >= 0) & (arr[:, 0] < W) &
                (arr[:, 1] >= 0) & (arr[:, 1] < H)
            )
            valid = arr[in_bounds]
            if len(valid) < 2:
                return None
            return valid.astype(np.int32)

        def _clip_segment(seg):
            """检查线段两端点是否在图像范围内，返回 int32 数组或 None。"""
            arr = _denorm(seg)
            if np.any(arr[:, 0] < 0) or np.any(arr[:, 0] >= W) or \
               np.any(arr[:, 1] < 0) or np.any(arr[:, 1] >= H):
                return None
            return arr.astype(np.int32)

        for room in rooms:
            verts = _clip_pts(room.get('vertices', []))
            if verts is not None:
                cv2.polylines(canvas, [verts], True, (0, 0, 255), 2)
            else:
                print(f"[draw_image] 房间 '{room.get('name', '?')}' 顶点全部越界或为空，跳过")

            # 每扇门是独立线段 [[start], [end]]，逐条绘制
            for seg in room.get('doors', []):
                pts = _clip_segment(seg)
                if pts is not None:
                    cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)

        return canvas

    def __getitem__(self, idx):
        # 边界检查：防止索引越界
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        
        item = self.data
        houseData = item['HouseData']
        floors = houseData['Floors']
    
        floor = floors[0]
        rooms = floor['Rooms'].copy()  # 复制列表，避免修改原数据导致循环
        floor_id = floor['ID']
        img_path = os.path.join(self.img_dir, floor_id + ".png")
        
        # 1. 加载 BEV 图像
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {img_path}")
        h, w = image.shape[:2]
        if w > self.TARGET_SIZE or h > self.TARGET_SIZE:
            raise ValueError(
                f"图像尺寸 ({w}x{h}) 超过限制 {self.TARGET_SIZE}×{self.TARGET_SIZE}，"
                f"路径: {img_path}"
            )
        scale = 1.0

        # 2. 创建 1024×1024 画布，原图居中贴入
        canvas = np.zeros((self.TARGET_SIZE, self.TARGET_SIZE, 3), dtype=np.uint8)
        canvas[:] = (0, 0, 0)
        if h > 0 and w > 0:
            pad_left = (self.TARGET_SIZE - w) // 2
            pad_top = (self.TARGET_SIZE - h) // 2
            canvas[pad_top : pad_top + h, pad_left : pad_left + w] = image

        # 画布中心固定为 (512, 512)，JSON 坐标以原图中心为原点，需乘 scale 后平移到画布
        cx, cy = self.TARGET_SIZE / 2.0, self.TARGET_SIZE / 2.0

        norm_w = float(self.TARGET_SIZE)
        norm_h = float(self.TARGET_SIZE)

        def to_topleft(x, y):
            """将以图像中心为原点的像素坐标转换为 1024×1024 画布的左上角为原点的像素坐标。"""
            return x * scale + cx, -y * scale + cy

        def to_norm(px, py):
            """像素坐标归一化到 [0, 1]，供模型训练使用。"""
            return px / norm_w, py / norm_h

        # 遍历每个房间，提取顶点和门信息
        processed_rooms = []
        for room in rooms:
            vertices = []
            room_doors = []
            room_name = room['Info']['Name']
            position = room['Info']['Position']
            position_x = position['x']
            position_y = position['z']

            walls = room.get('Walls', [])
            for wall in walls:
                sx, sy = to_topleft(wall['Start']['Up']['Position']['x'] + position_x,
                                    wall['Start']['Up']['Position']['z'] + position_y)
                ex, ey = to_topleft(wall['End']['Up']['Position']['x'] + position_x,
                                    wall['End']['Up']['Position']['z'] + position_y)
                vertices.append([*to_norm(sx, sy)])
                vertices.append([*to_norm(ex, ey)])

            doors = room.get('Doors', [])
            for door in doors:
                sx, sy = to_topleft(door['Start']['Up']['Position']['x'] + position_x,
                                    door['Start']['Up']['Position']['z'] + position_y)
                ex, ey = to_topleft(door['End']['Up']['Position']['x'] + position_x,
                                    door['End']['Up']['Position']['z'] + position_y)
                room_doors.append([[*to_norm(sx, sy)], [*to_norm(ex, ey)]])

            processed_rooms.append({
                'name': room_name,
                'vertices': vertices,   # 归一化坐标 [0,1]
                'doors': room_doors,     # 归一化坐标 [0,1]
                'img_shape': (self.TARGET_SIZE, self.TARGET_SIZE)
            })

        # ---------- 调试：输出各房间及全局外包围盒 ----------
        print(f"[DataLoader] floor={floor_id}  orig=({w}x{h})  canvas={self.TARGET_SIZE}x{self.TARGET_SIZE}  rooms={len(processed_rooms)}")
        all_pts = []
        for room in processed_rooms:
            # doors 现在是 List[[[sx,sy],[ex,ey]]]，展平后再合并
            door_pts = [pt for seg in room['doors'] for pt in seg]
            pts = room['vertices'] + door_pts
            if not pts:
                print(f"  [{room['name']}] 无顶点数据")
                continue
            arr = np.array(pts)
            mn, mx = arr.min(axis=0), arr.max(axis=0)
            print(f"  [{room['name']}]  x:[{mn[0]:.3f}, {mx[0]:.3f}]  y:[{mn[1]:.3f}, {mx[1]:.3f}] (norm)")
            all_pts.extend(pts)
        if all_pts:
            g = np.array(all_pts)
            gmn, gmx = g.min(axis=0), g.max(axis=0)
            print(f"  [全局bbox]   x:[{gmn[0]:.3f}, {gmx[0]:.3f}]  y:[{gmn[1]:.3f}, {gmx[1]:.3f}] (norm)")
        # -----------------------------------------------------
        
        return {
            'floor_id': floor_id,
            'img_path': img_path,
            'rooms': processed_rooms,
            'image': canvas,                    # 1024×1024 画布，原图居中贴入（可视化用）
            'bev_tensor': self.bev_image_to_tensor(canvas),  # [3,H,W] float32 [0,1]，模型输入
        }

if __name__ == "__main__":
    # 替换为你的实际路径
    json_path = "/Users/a58/Downloads/ae3a0c904df74464b19ac6d2943b9f6dtemps.177332101001628_138689714_2_24964851/Datas/ViewData.txt"
    img_dir = "/Users/a58/Downloads/ae3a0c904df74464b19ac6d2943b9f6dtemps.177332101001628_138689714_2_24964851/Datas/BasePlan/"
    
    # 初始化数据集
    dataset = RoomFormerV1Dataset(json_path=json_path, img_dir=img_dir)
    data_sample = dataset[0]
    json = data_sample['rooms']
    image = data_sample['image']
    draw_image = dataset.draw_image(image, json)
    cv2.imshow("draw_image", draw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    