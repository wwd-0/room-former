# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", poly_refine=True, return_intermediate_dec=False, aux_loss=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, query_pos_type="none",
                 use_pano_cross_attn=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points,
                                                          use_pano_cross_attn=use_pano_cross_attn)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, poly_refine, return_intermediate_dec, aux_loss, query_pos_type)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if query_pos_type == 'sine':
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, tgt=None, tgt_masks=None,
                pano_memory=None, pano_pos=None, pano_key_padding_mask=None,
                pano_token_uv=None, pano_tokens_per_pano=None,
                bev_world_meta=None, pose_matrices=None, pano_counts=None,
                pano_image_hw=None, use_pano_geom_bias=False, pano_sigma=64.0):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = query_embed.sigmoid()
        init_reference_out = reference_points

        # decoder
        hs, inter_references, inter_classes = self.decoder(
            tgt, reference_points, memory, src_flatten,
            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten, tgt_masks,
            pano_memory=pano_memory, pano_pos=pano_pos, pano_key_padding_mask=pano_key_padding_mask,
            pano_token_uv=pano_token_uv, pano_tokens_per_pano=pano_tokens_per_pano,
            bev_world_meta=bev_world_meta, pose_matrices=pose_matrices, pano_counts=pano_counts,
            pano_image_hw=pano_image_hw, use_pano_geom_bias=use_pano_geom_bias, pano_sigma=pano_sigma)

        return hs, init_reference_out, inter_references, inter_classes


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_pano_cross_attn=False):
        super().__init__()

        # cross attention (BEV)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # panorama cross attention (optional)
        self.pano_cross_attn = None
        if use_pano_cross_attn:
            self.pano_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.pano_dropout = nn.Dropout(dropout)
            self.pano_norm = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _pano_cross_attn_forward(self, tgt, query_pos, mem, pos, attn_bias):
        """Multi-head attention; optional additive attn_mask (equirectangular bias)."""
        mha = self.pano_cross_attn
        e_dim = tgt.shape[-1]
        n_head = mha.num_heads
        d_head = e_dim // n_head
        b, l_q, _ = tgt.shape
        s_k = mem.shape[1]
        q_p = self.with_pos_embed(tgt, query_pos)
        kv_p = self.with_pos_embed(mem, pos)
        w = mha.in_proj_weight
        b_proj = mha.in_proj_bias
        q = F.linear(q_p, w[:e_dim], b_proj[:e_dim] if b_proj is not None else None)
        k = F.linear(kv_p, w[e_dim:2 * e_dim], b_proj[e_dim:2 * e_dim] if b_proj is not None else None)
        v = F.linear(mem, w[2 * e_dim:], b_proj[2 * e_dim:] if b_proj is not None else None)
        q = q.view(b, l_q, n_head, d_head).transpose(1, 2)
        k = k.view(b, s_k, n_head, d_head).transpose(1, 2)
        v = v.view(b, s_k, n_head, d_head).transpose(1, 2)
        am = attn_bias
        if am is not None:
            am = am.unsqueeze(1).expand(-1, n_head, -1, -1)
        drop_p = self.pano_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=drop_p)
        out = out.transpose(1, 2).reshape(b, l_q, e_dim)
        return mha.out_proj(out)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                level_start_index, src_padding_mask=None, tgt_masks=None,
                pano_memory=None, pano_pos=None, pano_key_padding_mask=None,
                ref_xy01=None, pano_token_uv=None, pano_tokens_per_pano=None,
                bev_world_meta=None, pose_matrices=None, pano_counts=None,
                pano_image_hw=None, use_pano_geom_bias=False, pano_sigma=64.0):
        # 1) self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=tgt_masks)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 2) BEV cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 3) panorama cross attention (optional equirectangular back-project bias)
        if self.pano_cross_attn is not None and pano_memory is not None:
            attn_bias = None
            if (
                use_pano_geom_bias
                and ref_xy01 is not None
                and bev_world_meta is not None
                and pose_matrices is not None
                and pano_counts is not None
                and pano_token_uv is not None
                and pano_tokens_per_pano is not None
                and pano_image_hw is not None
            ):
                from util.pano_projector import (
                    bev_norm_xy_to_world_xyz,
                    world_to_pano_uv_torch,
                    build_pano_attn_bias_from_uv,
                )
                Hp, Wp = int(pano_image_hw[0]), int(pano_image_hw[1])
                world = bev_norm_xy_to_world_xyz(
                    ref_xy01,
                    bev_world_meta["origin_xz"],
                    tuple(bev_world_meta["grid_hw"]),
                    bev_world_meta["meters_per_pixel"],
                    bev_world_meta.get("world_y"),
                )
                proj_uv, _, _ = world_to_pano_uv_torch(
                    world, pose_matrices, Wp, Hp)
                attn_bias = build_pano_attn_bias_from_uv(
                    proj_uv,
                    pano_token_uv,
                    pano_counts,
                    int(pano_tokens_per_pano),
                    sigma_px=float(pano_sigma),
                    key_padding_mask=pano_key_padding_mask,
                )

            tgt2 = self._pano_cross_attn_forward(
                tgt, query_pos, pano_memory, pano_pos, attn_bias)
            tgt = tgt + self.pano_dropout(tgt2)
            tgt = self.pano_norm(tgt)

        # 4) ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, poly_refine=True, return_intermediate=False, aux_loss=False, query_pos_type='none'):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.poly_refine = poly_refine
        self.return_intermediate = return_intermediate
        self.aux_loss = aux_loss
        self.query_pos_type = query_pos_type
        
        self.coords_embed = None
        self.class_embed = None
        self.pos_trans = None
        self.pos_trans_norm = None

    def get_query_pos_embed(self, ref_points):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats) # [128]
        # N, L, 2
        ref_points = ref_points * scale
        # N, L, 2, 128
        pos = ref_points[:, :, :, None] / dim_t
        # N, L, 256
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(self, tgt, reference_points, src, src_flatten, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, tgt_masks=None,
                pano_memory=None, pano_pos=None, pano_key_padding_mask=None,
                pano_token_uv=None, pano_tokens_per_pano=None,
                bev_world_meta=None, pose_matrices=None, pano_counts=None,
                pano_image_hw=None, use_pano_geom_bias=False, pano_sigma=64.0):
        output = tgt    # [10, 800, 256]

        intermediate = []
        intermediate_reference_points = []
        intermediate_classes = []
        point_classes = torch.zeros(output.shape[:2]).unsqueeze(-1).to(output.device)
        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            
            if self.query_pos_type == 'sine':
                query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))

            elif self.query_pos_type == 'none':
                query_pos = None

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                           src_level_start_index, src_padding_mask, tgt_masks,
                           pano_memory=pano_memory, pano_pos=pano_pos,
                           pano_key_padding_mask=pano_key_padding_mask,
                           ref_xy01=reference_points,
                           pano_token_uv=pano_token_uv, pano_tokens_per_pano=pano_tokens_per_pano,
                           bev_world_meta=bev_world_meta, pose_matrices=pose_matrices,
                           pano_counts=pano_counts, pano_image_hw=pano_image_hw,
                           use_pano_geom_bias=use_pano_geom_bias, pano_sigma=pano_sigma)
    
            # iterative polygon refinement
            if self.poly_refine:
                offset = self.coords_embed[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points

            # if not using iterative polygon refinement, just output the reference points decoded from the last layer
            elif lid == len(self.layers)-1:
                offset = self.coords_embed[-1](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points
            
            # If aux loss supervision, we predict classes label from each layer and supervise loss
            if self.aux_loss:
                point_classes = self.class_embed[lid](output)
            # Otherwise, we only predict class label from the last layer
            elif lid == len(self.layers)-1:
                point_classes = self.class_embed[-1](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_classes.append(point_classes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_classes)

        return output, reference_points, point_classes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        poly_refine=args.with_poly_refine,
        return_intermediate_dec=True,
        aux_loss=args.aux_loss,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        query_pos_type=args.query_pos_type,
        use_pano_cross_attn=getattr(args, 'use_pano', False))


