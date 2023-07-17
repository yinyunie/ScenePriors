#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
from scipy import stats
import torch
from torch.nn import L1Loss, CrossEntropyLoss, BCEWithLogitsLoss, CosineSimilarity, BCELoss
import torch.distributions as dist
from models.registers import LOSSES
from net_utils.matcher_tracking import HungarianMatcher
from net_utils.box_utils import normalize_x1y1x2y2
from external.fast_transformers.fast_transformers.masking import LengthMask
from torch.nn import functional as F


class BaseLoss(object):
    '''base loss class'''
    def __init__(self, weight=1, cfg=None, device='cuda'):
        '''initialize loss module'''
        super(BaseLoss, self).__init__()
        self.weight = weight
        self.cfg = cfg
        self.device = device


@LOSSES.register_module
class Null(BaseLoss):
    '''This loss function is for modules where a loss preliminary calculated.'''
    def __call__(self, loss):
        return self.weight * torch.mean(loss)


@LOSSES.register_module
class KL(BaseLoss):
    def __init__(self, weight=1, cfg=None, device='cuda'):
        super(KL, self).__init__(weight=weight, cfg=cfg, device=device)
        self.z_dim = cfg.config.data.z_dim
        self.device = device

    def get_prior_z(self, z_dim, device):
        ''' Returns prior distribution for latent code z.
        Args:
            zdim: dimension of latent code z.
            device (device): pytorch device
        '''
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

        return p0_z

    def __call__(self, latent_dist):
        p0_z = self.get_prior_z(self.z_dim, self.device)
        kl = dist.kl_divergence(latent_dist, p0_z).sum(dim=-1)
        kl = kl.mean()
        return {'total': kl * self.weight, 'kl': kl}


@LOSSES.register_module
class MultiViewRenderLoss(BaseLoss):
    def __init__(self, weight=1, cfg=None, device='cuda'):
        super(MultiViewRenderLoss, self).__init__(weight=weight, cfg=cfg, device=device)
        self.l1_loss = L1Loss(reduction='none')
        self.cos_sim = CosineSimilarity(dim=-1)
        self.matcher = HungarianMatcher(1, 5)
        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.bce_loss = BCELoss(reduction='none')
        self.completeness_loss = BCEWithLogitsLoss(reduction='mean')

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def mask_iou(self, mask1, mask2):
        mask1 = mask1.flatten(-2, -1)
        mask2 = mask2.flatten(-2, -1)
        area1 = mask1.sum(dim=-1)
        area2 = mask2.sum(dim=-1)
        inter = torch.logical_and(mask1, mask2)
        inter = inter.sum(dim=-1)
        union = area1 + area2 - inter
        return inter / (union + 1e-5)

    def get_obj_weighted_loss(self, obj_view_loss, obj_view_mask):
        n_view_per_obj = obj_view_mask.sum(dim=-1)
        obj_view_loss = (obj_view_loss * obj_view_mask).sum(dim=-1) / (n_view_per_obj + 1e-6)
        return torch.mean(torch.masked_select(obj_view_loss, n_view_per_obj > 0))

    def frustum_loss(self, est_3d_center_rays, gt_2d_center_rays):
        frustum_loss = 1 - self.cos_sim(est_3d_center_rays, gt_2d_center_rays)
        return frustum_loss

    def get_frustum_loss(self, est_points_3d, gt_x1y1x2y2, batch_pred_idx, batch_gt_idx,
                         cam_Ks, cam_Ts, not_in_frustum_mask):
        n_view = cam_Ks.size(1)
        # get est rays
        est_points_3d = est_points_3d[batch_pred_idx].mean(dim=-2)
        est_points_3d = est_points_3d.unsqueeze(1).expand(-1, n_view, -1).contiguous()
        est_cam_Ts = cam_Ts[batch_pred_idx[0]]
        est_box3dcenter_rays = est_points_3d - est_cam_Ts[:, :, :3, 3]
        # get gt rays
        gt_boxes = gt_x1y1x2y2[batch_gt_idx]
        gt_box2dcenter = (gt_boxes[..., :2] + gt_boxes[..., 2:4]) / 2.

        gt_cam_Ks = cam_Ks[batch_gt_idx[0]]
        gt_cam_Ts = cam_Ts[batch_gt_idx[0]]

        inv_cam_Ks = 1. / torch.diagonal(gt_cam_Ks[..., :2, :2], dim1=-2, dim2=-1)
        gt_box2d_cam = inv_cam_Ks * (gt_box2dcenter - gt_cam_Ks[..., :2, 2])
        gt_box2d_cam = F.pad(gt_box2d_cam, (0, 1), "constant", 1)
        gt_box2d_cam[..., 1] *= -1
        gt_box2d_cam[..., 2] *= -1
        gt_box2dcenter_rays = torch.einsum('bvij,bvj->bvi', gt_cam_Ts[..., :3, :3], gt_box2d_cam)

        frustum_loss = self.frustum_loss(est_box3dcenter_rays, gt_box2dcenter_rays)
        frustum_loss = self.get_obj_weighted_loss(frustum_loss, not_in_frustum_mask)

        return frustum_loss

    def views_loss(self, est_data, gt_data, start_deform=False, return_matching=False):
        '''Calculate rendering loss.'''
        # indicates the instance marks for each object
        gt_obj_view_mask = gt_data['inst_marks']
        # indicates how many objects occur in each scene
        if (self.cfg.config.mode == 'demo' and self.cfg.config.data.n_views == 1) or (
                self.cfg.config.mode == 'test' and self.cfg.config.test.n_views_for_finetune == 1):
            obj_lens = self.cfg.max_n_obj * torch.ones_like(gt_data['max_len'][:, 0])
        else:
            obj_lens = gt_data['max_len'][:, 0]
        # indicates which est objects are used for loss calculation
        pred_mask = LengthMask(obj_lens).bool_matrix

        '''prepare est data'''
        est_points_2d = est_data['points_2d']
        est_cls_scores = est_data['classes_completeness'][..., :-1]
        in_frustum = est_data['in_frustum']

        '''prepare gt data'''
        gt_box2ds = gt_data['box2ds_tr']
        cam_Ks = gt_data['cam_K']
        cam_Ts = gt_data['cam_T']
        image_size = gt_data['image_size']

        gt_cls = gt_box2ds[..., 4:]
        gt_labels = gt_cls.argmax(dim=-1).max(dim=1)[0]
        gt_x1y1x2y2 = gt_box2ds[..., :4]

        '''bipartite matching'''
        gt_obj_view_mask = gt_obj_view_mask.transpose(1, 2)
        in_frustum = in_frustum.transpose(1, 2)
        est_points_2d = est_points_2d.transpose(1, 2)
        gt_x1y1x2y2 = gt_x1y1x2y2.transpose(1, 2)

        est_x1y1x2y2 = torch.cat([torch.min(est_points_2d, dim=-2)[0], torch.max(est_points_2d, dim=-2)[0]], dim=-1)

        normalized_est_x1y1x2y2 = normalize_x1y1x2y2(est_x1y1x2y2, image_size)
        normalized_gt_x1y1x2y2 = normalize_x1y1x2y2(gt_x1y1x2y2, image_size)

        pred = {'x1y1x2y2': normalized_est_x1y1x2y2, 'logits': est_cls_scores}
        gt = {'x1y1x2y2': normalized_gt_x1y1x2y2, 'cls': gt_labels}

        indices = self.matcher(pred, gt, pred_mask=pred_mask, gt_mask=gt_obj_view_mask)
        batch_pred_idx = self._get_src_permutation_idx(indices)
        batch_gt_idx = self._get_tgt_permutation_idx(indices)

        '''get in_frustum mask'''
        gt_obj_view_mask = gt_obj_view_mask[batch_gt_idx]
        in_frustum_mask = in_frustum[batch_pred_idx]

        '''calculate loss'''
        # frustum loss
        not_in_frustum_mask = torch.logical_and(torch.logical_not(in_frustum_mask), gt_obj_view_mask)
        if (False not in in_frustum_mask) or (True not in not_in_frustum_mask):
            frustum_loss = torch.tensor(0., device=self.device)
        else:
            # get est rays
            frustum_loss = self.get_frustum_loss(est_data['points_3d'], gt_x1y1x2y2, batch_pred_idx, batch_gt_idx, cam_Ks,
                                                 cam_Ts, not_in_frustum_mask)

        # semantic loss
        est_cls_scores = est_cls_scores[batch_pred_idx]
        gt_labels = gt_labels[batch_gt_idx]
        box_cls_loss = self.ce_loss(est_cls_scores, gt_labels)

        view_mask = torch.logical_and(in_frustum_mask, gt_obj_view_mask)
        if True not in view_mask:
            box_loss = torch.tensor(0., device=self.device)
            mask_loss = torch.tensor(0., device=self.device)

            losses = {'frustum_loss': frustum_loss,
                      'box_cls_loss': box_cls_loss,
                      'box_loss': box_loss,
                      'mask_loss': mask_loss}

            extra_output = {}
            if return_matching:
                extra_output['pred_gt_matching'] = indices

            return losses, extra_output

        # box loss
        normalized_est_x1y1x2y2 = normalized_est_x1y1x2y2[batch_pred_idx]
        normalized_gt_x1y1x2y2 = normalized_gt_x1y1x2y2[batch_gt_idx]
        box_loss = self.l1_loss(normalized_est_x1y1x2y2, normalized_gt_x1y1x2y2).sum(dim=-1)
        box_loss = self.get_obj_weighted_loss(box_loss, view_mask)

        if start_deform:
            '''mask loss'''
            # indicates the maximal object number in a batch.
            max_gt_obj_len = max(obj_lens)

            est_inst_masks = torch.cat(
                [(est_data['obj_ids'] == obj_id).unsqueeze(1) for obj_id in range(max_gt_obj_len)], dim=1)
            gt_inst_masks = torch.cat(
                [(gt_data['masks_tr'] == obj_id).unsqueeze(1) for obj_id in range(max_gt_obj_len)], dim=1)

            # mask loss
            gt_inst_masks = gt_inst_masks[batch_gt_idx]
            # get iou between est and gt masks
            est_inst_masks_indexed = est_inst_masks[batch_pred_idx]
            ious = self.mask_iou(est_inst_masks_indexed, gt_inst_masks)
            iou_mask = torch.logical_and(gt_obj_view_mask, (ious >= 0.5))
            if iou_mask.any():
                est_silhouettes = est_data['silhouettes'][:, None].expand(-1, max_gt_obj_len, -1, -1, -1)
                est_silhouettes = est_inst_masks.float() * est_silhouettes

                est_silhouettes = est_silhouettes[batch_pred_idx]
                mask_loss = self.bce_loss(est_silhouettes, gt_inst_masks.float())

                mask_loss = mask_loss.flatten(-2, -1)
                mask_loss = mask_loss.mean(dim=-1)
                mask_loss = torch.masked_select(mask_loss, iou_mask).mean()
            else:
                mask_loss = torch.tensor(0., device=self.device)
        else:
            mask_loss = torch.tensor(0., device=self.device)

        losses = {'frustum_loss': frustum_loss,
                  'box_cls_loss': box_cls_loss,
                  'box_loss': box_loss,
                  'mask_loss': mask_loss}

        extra_output = {}
        if return_matching:
            extra_output['pred_gt_matching'] = indices

        return losses, extra_output

    def __call__(self, est_data, gt_data, start_deform=False, return_matching=False, if_mask_loss=True, **kwargs):
        '''Calculate rendering loss'''
        view_losses, extra_output = self.views_loss(est_data, gt_data, start_deform,
                                                    return_matching=return_matching)

        # calculate completeness loss
        obj_lens = gt_data['max_len'][:, 0]
        n_batch = obj_lens.size(0)
        completeness_score = est_data['classes_completeness'][..., [-1]]
        gt_completeness = torch.zeros_like(completeness_score)
        gt_completeness[range(n_batch), obj_lens-1] = 1.
        gt_completeness = torch.cumsum(gt_completeness, dim=1)
        completeness_loss = self.completeness_loss(completeness_score, gt_completeness)
        if self.cfg.config.mode == 'test':
            completeness_loss = completeness_loss * (self.cfg.config['test'].n_views_for_finetune != 1)
        elif self.cfg.config.mode == 'demo':
            completeness_loss = completeness_loss * (self.cfg.config.data.n_views != 1)

        total_loss = view_losses['frustum_loss'] + view_losses['box_cls_loss'] + 5 * view_losses['box_loss'] + completeness_loss

        if start_deform:
            mask_loss = view_losses['mask_loss']
            if if_mask_loss == False:
                mask_loss = mask_loss * 0

            total_loss = total_loss + 2 * mask_loss

        return {'total': total_loss * self.weight, **view_losses,
                'completeness_loss': completeness_loss}, extra_output