# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

# 创建一个上三角矩阵：
#  1 1 1 1 1 1,.........1
#  1 1 1 1 1 1,.......1 0
#  1 1 1 1 1 1,.....1 0 0
#  ......
#  1 0 0 0 0 0 .........0

def get_mask(tscale):
    bm_mask = []
    for idx in range(tscale):
        mask_vector = [1 for i in range(tscale - idx)
                       ] + [0 for i in range(idx)]
        bm_mask.append(mask_vector)
    bm_mask = np.array(bm_mask, dtype=np.float32)
    return torch.Tensor(bm_mask)  # (100,100)的上三角矩阵

# pred_bm :(16,2,100,100)    gt_iou_map: (16,100,100)
# pred_start:(16,100)        gt_start:(16,100)
# pred_end:(16,100)          gt_end:(16,100)
# bm_mask: (100,100)

def bmn_loss_func(pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end, bm_mask):
    pred_bm_reg = pred_bm[:, 0].contiguous()  #(16,100,100)
    pred_bm_cls = pred_bm[:, 1].contiguous()  #(16,100,100)

    gt_iou_map = gt_iou_map * bm_mask  #(16,100,100)

    pem_reg_loss = pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)  # 0.0411
    pem_cls_loss = pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)  # 0.6933
    tem_loss = tem_loss_func(pred_start, pred_end, gt_start, gt_end)    # 1.3877

    loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
    return loss, tem_loss, pem_reg_loss, pem_cls_loss


def tem_loss_func(pred_start, pred_end, gt_start, gt_end):
    def bi_loss(pred_score, gt_label):  #(16,100),(16,100)
        pred_score = pred_score.view(-1)  # 1600
        gt_label = gt_label.view(-1)      # 1600
        pmask = (gt_label > 0.5).float()  
        num_entries = len(pmask)   # 1600
        num_positive = torch.sum(pmask)  # 77
        ratio = num_entries / num_positive  # 20.77
        coef_0 = 0.5 * ratio / (ratio - 1)   #  0.5253  
        coef_1 = 0.5 * ratio   # 10.3896
        epsilon = 0.000001  # 避免log陷入负无穷
        loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon)*(1.0 - pmask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

    loss_start = bi_loss(pred_start, gt_start)  # 0.6937
    loss_end = bi_loss(pred_end, gt_end)  # 0.6939
    loss = loss_start + loss_end 
    return loss


def pem_reg_loss_func(pred_score, gt_iou_map, mask):

    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
    u_lmask = u_lmask * mask

    num_h = torch.sum(u_hmask)  # 8136
    num_m = torch.sum(u_mmask)  # 32979
    num_l = torch.sum(u_lmask)  # 32900

    # 创建新的u_smmask和u_slmask为了平衡正负样本
    r_m = num_h / num_m  # 0.2473
    u_smmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()  
    u_smmask = u_mmask * u_smmask
    u_smmask = (u_smmask > (1. - r_m)).float()  # (16,100,100)

    r_l = num_h / num_l  # 0.2476
    u_slmask = torch.Tensor(np.random.rand(*gt_iou_map.shape)).cuda()
    u_slmask = u_lmask * u_slmask
    u_slmask = (u_slmask > (1. - r_l)).float()  # (16,100,100)

    weights = u_hmask + u_smmask + u_slmask  # (16,100,100)
    
    loss = F.mse_loss(pred_score* weights, gt_iou_map* weights)  
    loss = 0.5 * torch.sum(loss*torch.ones(*weights.shape).cuda()) / torch.sum(weights)


    return loss


def pem_cls_loss_func(pred_score, gt_iou_map, mask):  # pred_score = gt_iou_map = (16,100,100)

    pmask = (gt_iou_map > 0.9).float()
    nmask = (gt_iou_map <= 0.9).float()
    nmask = nmask * mask

    num_positive = torch.sum(pmask)  # 907  
    num_negtive = torch.sum(nmask)  # 79893
    num_entries = num_positive + torch.sum(nmask)  # 80800
    ratio = num_entries / num_positive  # 89.0849
    coef_0 = 0.5 * ratio / (ratio - 1)  # 0.5057
    coef_1 = 0.5 * ratio  # 44.5424
    epsilon = 0.000001
    loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask   # (16,100,100)
    loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * nmask  # (16,100,100)
    loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
    return loss
