import torch.utils.data

from .poly_data import build as build_poly


def build_dataset(image_set, args):
    if args.dataset_name in ['stru3d', 'scenecad', 'custom']:
        return build_poly(image_set, args)
    raise ValueError(f'dataset {args.dataset_name} not supported')
