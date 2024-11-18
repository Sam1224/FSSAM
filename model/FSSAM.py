import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import load_video_frames_from_data
from timm.models.layers import trunc_normal_


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def weighted_dice_loss(prediction, target_seg, weighted_val: float = 1.0, reduction: str = "sum", eps: float = 1e-8):
    target_seg = (target_seg == 1).float()  # B, H, W

    n, h, w = target_seg.shape

    prediction = prediction.reshape(-1, h, w)  # B, H, W
    target_seg = target_seg.reshape(-1, h, w)
    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h * w)  # B, H*W
    target_seg = target_seg.reshape(-1, h * w)

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum() / n
    elif reduction == "mean":
        loss = loss.mean()
    return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weighted_val: float = 1.0, reduction: str = "sum"):
        super(WeightedDiceLoss, self).__init__()
        self.weighted_val = weighted_val
        self.reduction = reduction

    def forward(self, prediction, target_seg):
        return weighted_dice_loss(prediction, target_seg, self.weighted_val, self.reduction)


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.dataset = args.data_set
        self.criterion = WeightedDiceLoss()
        self.print_freq = args.print_freq / 2
        self.pretrained = True
        self.classes = 1
        
        # ========================================
        # Build SAM2
        # ========================================
        self.sam2_weight = args.sam2_weight
        self.sam2_config = args.sam2_config
        self.sam2 = build_sam2_video_predictor(config_file=self.sam2_config, ckpt_path=self.sam2_weight, mode=None)

        # ========================================
        # Build DinoV2
        # ========================================
        try:
            ver_dino = args.ver_dino
        except:
            ver_dino = "dinov2_vitb14"
        if ver_dino == "dinov2_vitb14":
            self.num_layers = 12
        else:
            self.num_layers = 24
        # self.dino = torch.hub.load('dinov2/hub/facebookresearch_dinov2_main', ver_dino, trust_repo=True, source="local", verbose=False)
        self.dino = torch.hub.load('facebookresearch/dinov2', ver_dino, verbose=False)
        
        # ========================================
        # Mean and Std
        # ========================================
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

        self.num_refine = args.num_refine
        self.ver_refine = args.ver_refine

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.sam2.sam_mask_decoder.parameters()},
                {'params': model.sam2.memory_encoder.parameters()},
                {'params': model.sam2.memory_attention.parameters()},
            ], lr=LR, weight_decay=args.weight_decay
        )
        return optimizer

    def freeze_modules(self, model):
        for param in model.sam2.image_encoder.parameters():
            param.requires_grad = False
        for param in model.sam2.sam_prompt_encoder.parameters():
            param.requires_grad = False
        for param in model.sam2.obj_ptr_proj.parameters():
            param.requires_grad = False
        for param in model.sam2.mask_downsample.parameters():
            param.requires_grad = False
        for param in model.dino.parameters():
            param.requires_grad = False

    def cos_sim(self, query_feat_high, tmp_supp_feat, cosine_eps=1e-7):
        q = query_feat_high.flatten(2).transpose(-2, -1)
        s = tmp_supp_feat.flatten(2).transpose(-2, -1)

        tmp_query = q
        tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = (tmp_supp @ tmp_query) / (tmp_supp_norm @ tmp_query_norm + cosine_eps)
        return similarity

    def generate_prior_proto(self, query_feat_high, final_supp_list, mask_list, fts_size, normalize=False):
        bsize, ch_sz, sp_sz, _ = query_feat_high.size()[:]
        fg_list = []
        bg_list = []
        fg_sim_maxs = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            fg_supp_feat = Weighted_GAP(tmp_supp_feat, tmp_mask)
            bg_supp_feat = Weighted_GAP(tmp_supp_feat, 1 - tmp_mask)

            fg_sim = self.cos_sim(query_feat_high, fg_supp_feat, cosine_eps)
            bg_sim = self.cos_sim(query_feat_high, bg_supp_feat, cosine_eps)

            fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            bg_sim = bg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
            
            fg_sim_max = fg_sim.max(1)[0]  # bsize
            fg_sim_maxs.append(fg_sim_max.unsqueeze(-1))  # bsize, 1

            fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                        fg_sim.max(1)[0].unsqueeze(1) - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            bg_sim = (bg_sim - bg_sim.min(1)[0].unsqueeze(1)) / (
                    bg_sim.max(1)[0].unsqueeze(1) - bg_sim.min(1)[0].unsqueeze(1) + cosine_eps)

            fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)
            bg_sim = bg_sim.view(bsize, 1, sp_sz, sp_sz)

            fg_sim = F.interpolate(fg_sim, size=fts_size, mode='bilinear', align_corners=True)
            bg_sim = F.interpolate(bg_sim, size=fts_size, mode='bilinear', align_corners=True)
            fg_list.append(fg_sim)
            bg_list.append(bg_sim)
        fg_corr = torch.cat(fg_list, 1)  # bsize, shots, h, w
        bg_corr = torch.cat(bg_list, 1)
        corr = (fg_corr - bg_corr)
        corr[corr < 0] = 0
        corr_max = corr.view(bsize, len(final_supp_list), -1).max(-1)[0]  # bsize, shots

        if normalize:
            corr = corr.view(bsize, len(final_supp_list) * fts_size[0] * fts_size[1])
            corr = (corr - corr.min(1)[0].unsqueeze(1)) / (corr.max(1)[0].unsqueeze(1) - corr.min(1)[0].unsqueeze(1) + cosine_eps)
            corr = corr.view(bsize, len(final_supp_list), *fts_size)
        
        fg_sim_maxs = torch.cat(fg_sim_maxs, dim=-1)  # bsize, shots
        return fg_corr, bg_corr, corr, fg_sim_maxs, corr_max

    def generate_prior_mem(self, qry_feat, supp_fg, supp_mask, fts_size):
        bsize, ch_sz, sp_sz, _ = qry_feat.size()[:]
        cosine_eps = 1e-7

        fg_supp_feat = Weighted_GAP(supp_fg, supp_mask)  # b, c, 1, 1
        
        fg_sim = self.cos_sim(qry_feat, fg_supp_feat, cosine_eps)
        fg_sim = fg_sim.max(1)[0].view(bsize, sp_sz * sp_sz)
        fg_sim = (fg_sim - fg_sim.min(1)[0].unsqueeze(1)) / (
                    fg_sim.max(1)[0].unsqueeze(1) - fg_sim.min(1)[0].unsqueeze(1) + cosine_eps)
        fg_sim = fg_sim.view(bsize, 1, sp_sz, sp_sz)

        fg_sim = F.interpolate(fg_sim, size=fts_size, mode='bilinear', align_corners=True)
        
        return fg_sim

    def mem_refine(self, sup_fg_mem_feats, qry_fg_mem_feats, qry_ae_mem_feats, sup_fg, prior_fg, prior_ae, size, prior_mem_qry_sup_fg=None, cos_eps=1e-7):
        # qry-qry and qry-sup cosine similarities
        prior_mem_qry_sup_fg = prior_mem_qry_sup_fg if prior_mem_qry_sup_fg is not None else self.generate_prior_mem(qry_fg_mem_feats, sup_fg_mem_feats, sup_fg, size)
        prior_mem_qry_qry_fg = self.generate_prior_mem(qry_fg_mem_feats, qry_ae_mem_feats, prior_ae, size)
        prior_mem_fg = prior_mem_qry_qry_fg + (prior_mem_qry_sup_fg - 1.)
        prior_mem_fg[prior_mem_fg < 0] = 0

        # V1 - w/o weight norm
        # V2 - w/  weight norm
        # clipped prior norm
        if self.ver_refine == "v2":
            prior_mem_fg = rearrange(prior_mem_fg, 'b 1 h w -> b 1 (h w)')
            prior_mem_fg = (prior_mem_fg - prior_mem_fg.min(-1)[0].unsqueeze(-1)) / (prior_mem_fg.max(-1)[0].unsqueeze(-1) - prior_mem_fg.min(-1)[0].unsqueeze(-1) + cos_eps)
            prior_mem_fg = rearrange(prior_mem_fg, 'b 1 (h w) -> b 1 h w', h=size[0])
        
        # update ae mem
        qry_ae_mem_feats = qry_fg_mem_feats * prior_mem_fg + qry_ae_mem_feats * (1. - prior_mem_fg)

        # update ae prior
        prior_ae = prior_fg * prior_mem_fg + prior_ae * (1. - prior_mem_fg)
        prior_ae = rearrange(prior_ae, 'b 1 h w -> b 1 (h w)')
        prior_ae = (prior_ae - prior_ae.min(-1)[0].unsqueeze(-1)) / (prior_ae.max(-1)[0].unsqueeze(-1) - prior_ae.min(-1)[0].unsqueeze(-1) + cos_eps)
        prior_ae = rearrange(prior_ae, 'b 1 (h w) -> b 1 h w', h=size[0])

        return qry_ae_mem_feats, prior_ae, prior_mem_qry_sup_fg

    # que_img, sup_img, sup_mask, que_mask(meta), cat_idx(meta)
    @autocast()
    def forward(self, x, s_x, s_y, y_m, cat_idx=None, priors=None):
        b, _, h, w = x.size()  # b=1, 3, H, W
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_x = s_x.view(-1, 3, h, w)  # b*s, 3, 512, 512
            # remove padding (255)
            s_mask = s_y.clone().float()
            s_mask[s_mask == 255] = 0  # 1, shot, H, W

            # ========================================
            # DinoV2
            # ========================================
            if priors is None:
                with torch.no_grad():
                    dino_x = x
                    dino_s_x = s_x.view(-1, 3, h, w)
                    
                    # interpolate - 560
                    size = (560, 560)
                    dino_x = F.interpolate(dino_x, size=size, mode='bilinear', align_corners=True)
                    dino_s_x = F.interpolate(dino_s_x, size=size, mode='bilinear', align_corners=True)
                    
                    # normalize
                    dino_x = (dino_x - self.mean.to(x.device)) / self.std.to(x.device)
                    dino_s_x = (dino_s_x - self.mean.to(x.device)) / self.std.to(x.device)
                    
                    # dinov2 - feature extraction
                    qry_feats = self.dino.get_intermediate_layers(x=dino_x, n=range(0, self.num_layers), reshape=True)  # 12 feature maps, b, 768, 40, 40
                    sup_feats = self.dino.get_intermediate_layers(x=dino_s_x, n=range(0, self.num_layers), reshape=True)
                    
                    # only use last feature
                    qry_feat = qry_feats[-1]
                    sup_feat = sup_feats[-1]

                    size = qry_feat.size()[-2:]  # 40, 40
                    
                    # interpolate support mask
                    sup_mask = F.interpolate(s_mask.float(), size=size, mode='bilinear', align_corners=True)  # b, 1, 40, 40
                    
                    # generate dinov2 prior masks
                    size = (h, w)  # 512, 512
                    normalize = True
                    # prior
                    prior_fg, _, prior_ae, _, _ = self.generate_prior_proto(qry_feat, [sup_feat], [sup_mask], size, normalize=normalize)
                    prior_fg = F.interpolate(prior_fg, size=size, mode='bilinear', align_corners=True)  # b, 1, 512, 512
                    prior_ae = F.interpolate(prior_ae, size=size, mode='bilinear', align_corners=True)  # b, 1, 512, 512
                    priors = torch.cat([prior_fg, prior_ae], dim=1)  # b, 2, 512, 512 - for saving
            else:
                priors = priors  # b, 2, 512, 512
                prior_fg = priors[:, 0, ...].unsqueeze(1)  # b, 1, 512, 512
                prior_ae = priors[:, 1, ...].unsqueeze(1)  # b, 1, 512, 512

            # ========================================
            # SAM2 - video mode
            # ========================================
            x = load_video_frames_from_data(x, offload_video_to_cpu=False)  # b, 3, 512, 512
            s_x = load_video_frames_from_data(s_x, offload_video_to_cpu=False)  # b*s, 3, 512, 512
            with torch.no_grad():
                # obtain query and support features
                _, _, qry_feats, qry_poss, qry_sizes = self.sam2.get_image_feature_batch(x)
                _, _, sup_feats, sup_poss, sup_sizes = self.sam2.get_image_feature_batch(s_x)

            # add support prompt - gt mask
            sup_fg = s_mask[:, 0, ...].unsqueeze(1)  # b, 1, h, w
            (sup_fg_preds, sup_fg_obj_ptrs, sup_fg_mem_feats, sup_fg_mem_poss) = self.sam2.add_new_mask_batch(sup_feats, sup_sizes, sup_fg)  # support fg gt memory, b, 64, 32, 32

            # add query prompt - prior mask
            (qry_fg_preds, qry_fg_obj_ptrs, qry_fg_mem_feats, qry_fg_mem_poss) = self.sam2.add_new_mask_batch(qry_feats, qry_sizes, prior_fg)  # query fg prior memory, b, 64, 32, 32
            (qry_ae_preds, qry_ae_obj_ptrs, qry_ae_mem_feats, qry_ae_mem_poss) = self.sam2.add_new_mask_batch(qry_feats, qry_sizes, prior_ae)  # query ae prior memory, b, 64, 32, 32

            # ========================================
            # iterative memory refinement
            # ========================================
            sup_fg = F.interpolate(sup_fg, size=qry_sizes[-1], mode='bilinear', align_corners=True)
            prior_ae = F.interpolate(prior_ae, size=qry_sizes[-1], mode='bilinear', align_corners=True)
            prior_fg = F.interpolate(prior_fg, size=qry_sizes[-1], mode='bilinear', align_corners=True)

            num_refine = self.num_refine
            prior_mem_qry_sup_fg = None
            for i in range(num_refine):
                qry_ae_mem_feats, prior_ae, prior_mem_qry_sup_fg = self.mem_refine(
                    sup_fg_mem_feats, qry_fg_mem_feats, qry_ae_mem_feats,
                    sup_fg, prior_fg, prior_ae,
                    size=qry_sizes[-1],
                    prior_mem_qry_sup_fg=prior_mem_qry_sup_fg
                )

            # ========================================
            # support-calibrated memory attention
            # ========================================
            # propagate prompted frames
            sup_mask = F.interpolate(s_mask[:, 0, ...].unsqueeze(1).float(), size=qry_sizes[-1], mode='nearest')
            output_query, weights = self.sam2.propagate_in_video_batch_final(
                qry_feats, qry_poss, qry_sizes,
                sup_fg_mem_feats, sup_fg_mem_poss, sup_mask, sup_fg_preds, sup_fg_obj_ptrs,
                qry_ae_mem_feats, qry_ae_mem_poss, qry_ae_preds, qry_ae_obj_ptrs
            )
            output_query = output_query.squeeze(1)

            # Loss
            if self.training:
                main_loss = self.criterion(output_query, y_m.float())
                aux_loss1 = torch.zeros_like(main_loss)
                aux_loss2 = torch.zeros_like(main_loss)
                return output_query, main_loss, aux_loss1, aux_loss2
            else:
                return output_query, priors
