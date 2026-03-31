from pathlib import Path

import torch
import torch.utils.data

from pycocotools.coco import COCO
from PIL import Image
import cv2

from util.poly_ops import resort_corners
from detectron2.data import transforms as T
from torch.utils.data import Dataset
import numpy as np
import json
import os
from copy import deepcopy

from detectron2.data.detection_utils import annotations_to_instances, transform_instance_annotations
from detectron2.structures import BoxMode


class MultiPoly(Dataset):
    def __init__(self, img_folder, ann_file, transforms, semantic_classes,
                 pano_dir=None, pano_h=256, pano_w=512, max_pano=10,
                 bev_channels=3, use_depth=False):
        super(MultiPoly, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.bev_channels = bev_channels
        self.prepare = ConvertToCocoDict(self.root, self._transforms,
                                         bev_channels=bev_channels)

        self.pano_dir = pano_dir
        self.pano_h = pano_h
        self.pano_w = pano_w
        self.max_pano = max_pano
        self.use_depth = use_depth

    def get_image(self, path):
        return Image.open(os.path.join(self.root, path))

    def _load_panoramas(self, scene_dir):
        """Load panorama images, depth maps, and poses from viewData.txt.

        Expected directory layout (dataset_v1 format):
            {scene_dir}/
                viewData.txt           # JSON with HouseData.ShootSpots
                panoramas/玄关.jpg     # panorama RGB
                depths/玄关_depth.png  # depth map
        """
        if not os.path.isdir(scene_dir):
            return None, None, None

        viewdata_path = None
        for name in ('viewData.txt', 'ViewData.txt'):
            p = os.path.join(scene_dir, name)
            if os.path.isfile(p):
                viewdata_path = p
                break
        if viewdata_path is None:
            return None, None, None

        with open(viewdata_path, 'r') as f:
            viewdata = json.load(f)

        shoot_spots = viewdata.get('HouseData', {}).get('ShootSpots', [])
        if not shoot_spots:
            return None, None, None

        from .pose_utils import parse_shoot_spots, compute_pose_matrices
        parsed_spots = parse_shoot_spots(
            shoot_spots, base_dir=scene_dir,
            pano_subdir='panoramas', depth_subdir='depths')
        matrices = compute_pose_matrices(parsed_spots)

        pano_images = []
        depth_images = []
        pose_matrices = []
        for spot, mat in zip(parsed_spots, matrices):
            pano_path = spot['panorama_path']
            if not os.path.isfile(pano_path):
                continue
            img = Image.open(pano_path).convert('RGB')
            img = img.resize((self.pano_w, self.pano_h), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # (3, Hp, Wp)
            pano_images.append(torch.from_numpy(arr))
            pose_matrices.append(torch.from_numpy(mat))

            if self.use_depth:
                depth_path = spot.get('depth_path', '')
                if os.path.isfile(depth_path):
                    d_img = Image.open(depth_path)
                    d_img = d_img.resize((self.pano_w, self.pano_h), Image.NEAREST)
                    d_arr = np.array(d_img, dtype=np.float32)
                    if d_arr.ndim == 3:
                        d_arr = d_arr[:, :, 0]
                    d_arr = d_arr / d_arr.max().clip(min=1e-6)  # normalize to [0, 1]
                    depth_images.append(torch.from_numpy(d_arr).unsqueeze(0))  # (1, Hp, Wp)
                else:
                    depth_images.append(torch.zeros(1, self.pano_h, self.pano_w))

            if len(pano_images) >= self.max_pano:
                break

        if not pano_images:
            return None, None, None

        return pano_images, pose_matrices, (depth_images if self.use_depth else None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        if self.semantic_classes == -1:
            target = [t for t in target if t['category_id'] not in [16, 17, 18]]
        elif self.semantic_classes == 4:
            # 在线归并 4 类 (0:Room, 1:Door, 2:Window, 3:Hole)
            for t in target:
                orig_id = t['category_id']
                if orig_id < 16:
                    t['category_id'] = 0
                elif orig_id == 16:
                    t['category_id'] = 1
                elif orig_id == 17:
                    t['category_id'] = 2
                elif orig_id == 18:
                    t['category_id'] = 3

        path = coco.loadImgs(img_id)[0]['file_name']

        record = self.prepare(img_id, path, target)

        if self.pano_dir is not None:
            scene_key = str(Path(path).parent)
            scene_dir = os.path.join(self.root, scene_key)
            pano_imgs, pose_mats, depth_imgs = self._load_panoramas(scene_dir)
            record['pano_images'] = pano_imgs        # list of (3, Hp, Wp) or None
            record['pose_matrices'] = pose_mats      # list of (4, 4) or None
            record['depth_images'] = depth_imgs      # list of (1, Hp, Wp) or None
            record['pano_count'] = len(pano_imgs) if pano_imgs else 0

        # BEV 世界系 footprint，与 generate_bev.py / generate_pointcloud 反投影一致（可选）
        scene_key = str(Path(path).parent)
        scene_dir = os.path.join(self.root, scene_key)
        bev_meta_path = os.path.join(scene_dir, 'bev_meta.json')
        if os.path.isfile(bev_meta_path):
            with open(bev_meta_path, 'r', encoding='utf-8') as bf:
                bj = json.load(bf)
            record['bev_world_meta'] = {
                'origin_xz': torch.tensor(
                    [float(bj['origin_x']), float(bj['origin_z'])],
                    dtype=torch.float32),
                'meters_per_pixel': float(bj.get('meters_per_pixel', 0.02)),
                'world_y': float(bj.get('world_y', 0.0)),
            }

        return record


class ConvertToCocoDict(object):
    def __init__(self, root, augmentations, bev_channels=3):
        self.root = root
        self.augmentations = augmentations
        self.bev_channels = bev_channels

    def _load_image(self, file_name):
        if self.bev_channels == 3:
            img = np.array(Image.open(file_name).convert('RGB'))  # (H, W, 3)
            h, w = img.shape[:2]
            return img, h, w
        elif self.bev_channels == 1:
            img = np.array(Image.open(file_name).convert('L'))    # (H, W)
            h, w = img.shape
            return img, h, w
        else:
            img = np.array(Image.open(file_name))
            if img.ndim == 3:
                h, w = img.shape[:2]
            else:
                h, w = img.shape
            return img, h, w

    def _to_tensor(self, img):
        if img.ndim == 2:
            return (1/255) * torch.as_tensor(
                np.ascontiguousarray(np.expand_dims(img, 0)), dtype=torch.float32)
        else:
            return (1/255) * torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1)), dtype=torch.float32)

    def __call__(self, img_id, path, target):

        file_name = os.path.join(self.root, path)
        img, h, w = self._load_image(file_name)

        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record['image_id'] = img_id
        
        for obj in target: obj["bbox_mode"] = BoxMode.XYWH_ABS

        record['annotations'] = target

        if self.augmentations is None:
            record['image'] = self._to_tensor(img)
            record['instances'] = annotations_to_instances(target, (h, w), mask_format="polygon")
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record['image'] = self._to_tensor(image)
            
            annos = [
                transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                    )
                    for obj in record.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                    ]
            for anno in annos:
                anno['segmentation'][0] = resort_corners(anno['segmentation'][0])

            record['instances'] = annotations_to_instances(annos, (h, w), mask_format="polygon")
            
        return record

def make_poly_transforms(image_set):

    if image_set == 'train':
        return T.AugmentationList([
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomRotation([0.0, 90.0, 180.0, 270.0], expand=False, center=None, sample_style="choice")
            ]) 
        
    if image_set == 'val' or image_set == 'test':
        return None

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.dataset_root)
    assert root.exists(), f'provided data path {root} does not exist'

    PATHS = {
        "train": (root / "train", root / "annotations" / 'train.json'),
        "val": (root / "val", root / "annotations" / 'val.json'),
        "test": (root / "test", root / "annotations" / 'test.json')
    }

    img_folder, ann_file = PATHS[image_set]

    use_pano = getattr(args, 'use_pano', False)
    pano_dir = getattr(args, 'pano_dir', None)
    if use_pano and pano_dir is None:
        pano_dir = str(img_folder)
    pano_h = getattr(args, 'pano_input_h', 256)
    pano_w = getattr(args, 'pano_input_w', 512)
    max_pano = getattr(args, 'max_pano_count', 10)
    bev_channels = getattr(args, 'bev_channels', 3)
    use_depth = getattr(args, 'use_depth', False)

    dataset = MultiPoly(
        img_folder, ann_file,
        transforms=make_poly_transforms(image_set),
        semantic_classes=args.semantic_classes,
        pano_dir=pano_dir,
        pano_h=pano_h,
        pano_w=pano_w,
        max_pano=max_pano,
        bev_channels=bev_channels,
        use_depth=use_depth,
    )

    return dataset
