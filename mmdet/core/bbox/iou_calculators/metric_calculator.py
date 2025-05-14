import math
import torch

from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class BboxDistanceMetric(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6, weight=2, constant=99.5):
    assert mode in ['iou', 'iof', 'giou', 'wd', 'kl','center_distance2','exp_kl','kl_10','gcd','cwd','gjsd','nwd','cwdv2'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    if mode == 'center_distance2':
        # box1 , box2: x1, y1, x2, y2
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + 1e-6 #

        #distance = torch.sqrt(center_distance2)
    
        return center_distance2

    if mode == 'kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(1+kl)

        return kld

    if mode == 'kl_10':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(10+kl)

        return kld

    if mode == 'exp_kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = torch.exp(-kl/10)

        return kld

    if mode == 'wd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wassersteins = torch.sqrt(center_distance + wh_distance)

        normalized_wasserstein = torch.exp(-wassersteins/constant)

        return normalized_wasserstein
    
    if mode == 'nwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
        normalize_wasserstein = torch.sqrt(center_distance + wh_distance) / (torch.pow((w1 * w2 * h1 * h2), 1/4) + eps)

        nwd = 1 / (1 + normalize_wasserstein)

        return nwd
    
    if mode == 'gcd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        center_distance1 = (whs[..., 0] / w1) ** 2 + (whs[..., 1] / h1) ** 2
        wh_distance1 = (((w1 - w2) / w2) ** 2 + ((h1 - h2) / h2) ** 2) / 4

        center_distance2 = (whs[..., 0] / w2) ** 2 + (whs[..., 1] / h2) ** 2
        wh_distance2 = (((w1 - w2) / w1) ** 2 + ((h1 - h2) / h1) ** 2) / 4

        gcd_2 = (center_distance1 + wh_distance1 + center_distance2 + wh_distance2) / 2 

        gcd = torch.exp(-torch.sqrt(gcd_2))
        # gcd = 1 / (1 + torch.sqrt(gcd_2))
        
        return gcd
    
    if mode == 'cwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        center_distance = (whs[..., 0] ** 2 / (w1 * w2)) + (whs[..., 1] ** 2 / (h1 * h2))  + eps #
        wh_distance = (((w1 - w2) ** 2 / (w1 * w2))  + ((h1 - h2)  ** 2 / (h1 * h2))) / 4

        cwd_2 = center_distance + wh_distance

        cwd = torch.exp(-torch.sqrt(cwd_2))
        # cwd = 1 / (1 + cwd_2)

        return cwd

    if mode == 'cwdv2':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        center_distance = (whs[..., 0] ** 2 / (w1 * h2)) + (whs[..., 1] ** 2 / (h1 * w2)) #
        wh_distance = (((w1 - w2) ** 2 / (h1 * w2)) + ((h1 - h2)  ** 2 / (h2 * w1))) / 4

        cwd_2 = center_distance + wh_distance

        cwd = torch.exp(-torch.sqrt(cwd_2))
        # cwd = 1 / (1 + cwd_2)
        return cwd
    
    if mode == 'gjsd':
        g_bboxes1 = xy_wh_r_2_xy_sigma(bboxes1)
        g_bboxes2 = xy_wh_r_2_xy_sigma(bboxes2)
        gjsd = 1 / (1 + get_gjsd(g_bboxes1,g_bboxes2))

        return gjsd   
    
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if mode in ['box1_box2']:
        box1_box2 = area1[...,None] / area2[None,...]
        return box1_box2

    lt = torch.max(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap + eps

    if mode in ['giou']:
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])
        

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    
    if mode in ['iou', 'iof']:
        return ious

    # calculate gious
    if mode in ['giou']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

    if mode == 'giou':
        return gious

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def get_gjsd(pred, target, alpha=0.5):
    xy_p, Sigma_p = pred  # mu_1, sigma_1
    xy_t, Sigma_t = target # mu_2, sigma_2

    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)
    
    
    xy_p = xy_p[...,:,None,:2]
    xy_t = xy_t[...,None,:,:2]

    # get the inverse of Sigma_p and Sigma_t
    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)
    Sigma_t_inv = torch.stack((Sigma_t[..., 1, 1], -Sigma_t[..., 0, 1],
                               -Sigma_t[..., 1, 0], Sigma_t[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_t_inv = Sigma_t_inv / Sigma_t.det().unsqueeze(-1).unsqueeze(-1)

    Sigma_p = Sigma_p[...,:,None,:2,:2]
    Sigma_p_inv = Sigma_p_inv[...,:,None,:2,:2]
    Sigma_t = Sigma_t[...,None,:,:2,:2]
    Sigma_t_inv = Sigma_t_inv[...,None,:,:2,:2]
    
    Sigma_alpha_ori = ((1-alpha)*Sigma_p_inv + alpha*Sigma_t_inv)

    # get the inverse of Sigma_alpha_ori, namely Sigma_alpha
    Sigma_alpha =  torch.stack((Sigma_alpha_ori[..., 1, 1], -Sigma_alpha_ori[..., 0, 1],
                               -Sigma_alpha_ori[..., 1, 0], Sigma_alpha_ori[..., 0, 0]),
                              dim=-1).reshape(Sigma_alpha_ori.size(0), Sigma_alpha_ori.size(1), 2, 2)
    Sigma_alpha = Sigma_alpha / Sigma_alpha_ori.det().unsqueeze(-1).unsqueeze(-1)
    # get the inverse of Sigma_alpha, namely Sigma_alpha_inv
    Sigma_alpha_inv = torch.stack((Sigma_alpha[..., 1, 1], -Sigma_alpha[..., 0, 1],
                               -Sigma_alpha[..., 1, 0], Sigma_alpha[..., 0, 0]),
                              dim=-1).reshape(Sigma_alpha.size(0),Sigma_alpha.size(1), 2, 2)
    Sigma_alpha_inv = Sigma_alpha_inv / Sigma_alpha.det().unsqueeze(-1).unsqueeze(-1)

    # mu_alpha
    xy_p = xy_p.unsqueeze(-1)
    xy_t = xy_t.unsqueeze(-1)
    
    mu_alpha_1 = (1-alpha)* Sigma_p_inv.matmul(xy_p) + alpha * Sigma_t_inv.matmul(xy_t)
    mu_alpha = Sigma_alpha.matmul(mu_alpha_1)
    
    # the first part of GJSD 
    first_part = (1-alpha) * xy_p.permute(0,1,3,2).matmul(Sigma_p_inv).matmul(xy_p) + alpha * xy_t.permute(0,1,3,2).matmul(Sigma_t_inv).matmul(xy_t) - mu_alpha.permute(0,1,3,2).matmul(Sigma_alpha_inv).matmul(mu_alpha)
    second_part = ((Sigma_p.det() ** (1-alpha))*(Sigma_t.det() ** alpha))/(Sigma_alpha.det())
    second_part = second_part.log()

    if first_part.is_cuda:
        gjsd = 0.5 * (first_part.half().squeeze(-1).squeeze(-1) + second_part.half())
        #distance = 1/(1+gjsd)
    else:
        gjsd = 0.5 * (first_part.squeeze(-1).squeeze(-1) + second_part)
        #distance = 1/(1+gjsd)

    return gjsd