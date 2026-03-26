"""
WandB Visualization Utilities for RoomFormer training.

Provides:
  - log_prediction_images(): render GT vs Pred room polygons side-by-side
    and upload as a WandB Image grid every N epochs.
  - log_loss_curves(): rich metric logging helper (wraps existing dicts).
"""

import io
import math
import numpy as np
import torch
import wandb

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ─────────────────────────────────────────────
#  Color palette for up to 20 predicted rooms
# ─────────────────────────────────────────────
_PALETTE = [
    (255, 100, 100), (100, 200, 100), (100, 150, 255), (255, 200,  50),
    (200,  80, 200), ( 50, 220, 220), (255, 140,  50), (120, 255, 120),
    (255,  80, 180), ( 80, 180, 255), (200, 200,  60), (160, 100, 200),
    ( 60, 200, 160), (255, 160,  60), (100, 100, 200), (200, 100, 100),
    (100, 200, 200), (200, 160, 100), (160, 200, 100), (100, 160, 200),
]


def _make_canvas(bev_tensor: torch.Tensor, scale: int = 256) -> np.ndarray:
    """Convert BEV tensor (C, H, W) → (scale, scale, 3) uint8 BGR canvas."""
    img = bev_tensor.cpu().float()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img[:3]                     # take first 3 channels
    arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[:2] != (scale, scale):
        if HAS_CV2:
            arr = cv2.resize(arr, (scale, scale))
        else:
            # simple nearest-neighbor via numpy
            h, w = arr.shape[:2]
            ys = (np.arange(scale) * h // scale).astype(int)
            xs = (np.arange(scale) * w // scale).astype(int)
            arr = arr[np.ix_(ys, xs)]
    # Convert RGB → BGR for cv2 drawing; we'll flip back at the end
    return arr.copy()


def _draw_polys(canvas: np.ndarray, polys, color, thickness=2, fill_alpha=0.15):
    """Draw filled + outlined polygons on canvas in-place."""
    if not HAS_CV2:
        return canvas
    overlay = canvas.copy()
    for pts in polys:
        pts = np.array(pts, dtype=np.int32)
        if len(pts) < 3:
            continue
        bgr = color[::-1]   # RGB → BGR for cv2
        cv2.fillPoly(overlay, [pts], bgr)
        cv2.polylines(canvas, [pts], isClosed=True, color=bgr, thickness=thickness)
        # draw corner dots
        for p in pts:
            cv2.circle(canvas, tuple(p.tolist()), 3, bgr, -1)
    cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)
    return canvas


def _extract_pred_polys(outputs, sample_idx: int, num_polys: int, min_area: float = 100.0):
    """Extract predicted room polygons from model output for one scene."""
    pred_logits = outputs['pred_logits']   # (B, P, Q, 1)
    pred_coords = outputs['pred_coords']   # (B, P, Q, 2)

    fg_mask = torch.sigmoid(pred_logits[sample_idx]) > 0.5  # (P, Q, 1) or (P, Q)
    if fg_mask.dim() == 3:
        fg_mask = fg_mask.squeeze(-1)

    room_polys = []
    for j in range(fg_mask.shape[0]):
        valid = fg_mask[j]                          # (Q,)
        corners = pred_coords[sample_idx, j][valid]  # (K, 2)
        if len(corners) < 3:
            continue
        pts = (corners.cpu().float() * 255).numpy().astype(np.int32)
        if HAS_SHAPELY and len(pts) >= 3:
            try:
                if Polygon(pts).area < min_area:
                    continue
            except Exception:
                pass
        room_polys.append(pts)
    return room_polys


def _extract_gt_polys(gt_instance):
    """Extract GT polygons from a detectron2 Instances object."""
    polys = []
    for poly_list in gt_instance.gt_masks.polygons:
        pts = poly_list[0].reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def render_comparison(
    bev_tensor: torch.Tensor,
    gt_polys,
    pred_polys,
    scene_id,
    canvas_size: int = 256,
) -> np.ndarray:
    """
    Returns a (canvas_size, canvas_size*3, 3) uint8 RGB image:
      [BEV Map | GT polygons | Predicted polygons]
    """
    bev_img = _make_canvas(bev_tensor, canvas_size)

    # GT panel: white background
    gt_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240
    for i, pts in enumerate(gt_polys):
        _draw_polys(gt_canvas, [pts], _PALETTE[i % len(_PALETTE)], thickness=2)

    # Pred panel: white background
    pred_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240
    for i, pts in enumerate(pred_polys):
        _draw_polys(pred_canvas, [pts], _PALETTE[i % len(_PALETTE)], thickness=2)

    # Add text labels if cv2 available
    if HAS_CV2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bev_img,     f'BEV #{scene_id}', (4, 18), font, 0.45, (220,220,220), 1)
        cv2.putText(gt_canvas,   f'GT  #{scene_id} ({len(gt_polys)} rooms)', (4, 18), font, 0.4, ( 50, 50, 50), 1)
        cv2.putText(pred_canvas, f'Pred ({len(pred_polys)} rooms)', (4, 18), font, 0.45, ( 50, 50, 50), 1)

    # Convert BGR → RGB for wandb
    if HAS_CV2:
        bev_img   = cv2.cvtColor(bev_img,   cv2.COLOR_BGR2RGB)
        gt_canvas = cv2.cvtColor(gt_canvas, cv2.COLOR_BGR2RGB)
        pred_canvas = cv2.cvtColor(pred_canvas, cv2.COLOR_BGR2RGB)

    row = np.concatenate([bev_img, gt_canvas, pred_canvas], axis=1)
    return row


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

@torch.no_grad()
def log_prediction_images(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    num_polys: int,
    canvas_size: int = 256,
    max_scenes: int = 4,
    wandb_key: str = "viz/predictions",
):
    """
    Run the model on the first `max_scenes` validation scenes, render
    GT vs Pred side-by-side, and upload to WandB.

    Call this from your training loop every N epochs.

    Args:
        model:       the RoomFormer model (will be set to eval() then back)
        data_loader: validation DataLoader
        device:      torch device
        epoch:       current epoch number (shown in the WandB panel caption)
        num_polys:   args.num_polys
        canvas_size: pixel size of each panel (default 256)
        max_scenes:  how many scenes to visualise (default 4)
        wandb_key:   WandB logging key (default "viz/predictions")
    """
    from util.poly_ops import pad_gt_polys
    from engine import _prepare_pano_batch, _prepare_bev_world_meta

    was_training = model.training
    model.eval()

    panel_rows = []
    scene_count = 0

    for batched_inputs in data_loader:
        if scene_count >= max_scenes:
            break

        samples = [x["image"].to(device) for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        scene_ids = [x.get("image_id", scene_count + i) for i, x in enumerate(batched_inputs)]

        pano_images, pose_matrices, pano_counts, depth_images = _prepare_pano_batch(batched_inputs, device)
        bev_hw = (samples[0].shape[-2], samples[0].shape[-1])
        bev_world_meta = _prepare_bev_world_meta(batched_inputs, bev_hw, device)

        outputs = model(
            samples,
            pano_images=pano_images,
            pose_matrices=pose_matrices,
            pano_counts=pano_counts,
            depth_images=depth_images,
            bev_world_meta=bev_world_meta,
        )

        for i in range(len(samples)):
            if scene_count >= max_scenes:
                break

            gt_polys   = _extract_gt_polys(gt_instances[i])
            pred_polys = _extract_pred_polys(outputs, i, num_polys)
            row = render_comparison(
                samples[i],
                gt_polys,
                pred_polys,
                scene_id=scene_ids[i],
                canvas_size=canvas_size,
            )
            panel_rows.append(row)
            scene_count += 1

    if model.training != was_training:
        model.train(was_training)

    if not panel_rows:
        return

    # Stack all rows vertically into one tall image
    grid = np.concatenate(panel_rows, axis=0)
    caption = (
        f"Epoch {epoch} | Left: BEV map | Mid: GT rooms | Right: Predicted rooms\n"
        f"({scene_count} scenes shown)"
    )
    wandb.log({wandb_key: wandb.Image(grid, caption=caption)}, step=epoch)
