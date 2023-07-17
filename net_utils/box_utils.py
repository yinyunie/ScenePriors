#  Copyright (c) 2.2022. Yinyu Nie
#  License: MIT
import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def get_area(x1y1x2y2):
    return torch.prod((x1y1x2y2[..., 2:4] - x1y1x2y2[..., :2]), dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = get_area(boxes1)
    area2 = get_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def mask_iou(mask1, mask2):
    mask1 = mask1.flatten(-2, -1)
    mask2 = mask2.flatten(-2, -1)

    area1 = mask1.sum(dim=-1)
    area2 = mask2.sum(dim=-1)

    n_batch_1 = mask1.size(0)
    n_batch_2 = mask2.size(0)

    mask1_ext = mask1[:, None].expand(-1, n_batch_2, -1)
    mask2_ext = mask2[None].expand(n_batch_1, -1, -1)

    inter = torch.logical_and(mask1_ext, mask2_ext)
    inter = inter.sum(dim=-1)

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def pitch_yaw_roll_from_R(cam_R):
    ''' (torch version of pitch_yaw_roll_from_R in utils/tool.py)
    Get the pitch (x-axis), yaw (y-axis), roll (z-axis) angle from the camera rotation matrix.
      /|\ y
       |    / x axis
       |  /
       |/-----------> z axis
    pitch is the angle rotates along x axis.
    yaw is the angle rotates along y axis.
    roll is the angle rotates along z axis.
    :param cam_R: Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the left, up,
    forward vector relative to the world system.
    and R = R_z(roll)Ry_(yaw)Rx_(pitch)
    refer to: https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw
    '''
    if len(cam_R.shape) == 2:
        cam_R = cam_R[None]
    # in normal cases (capturing in a scene)
    # pitch ranges from [-pi/2, pi/2]
    # yaw ranges from [-pi, pi]
    # roll ranges from [-pi/2, pi/2]
    pitch = torch.arctan(cam_R[:, 2, 1]/cam_R[:, 2, 2])
    yaw = torch.arcsin(-cam_R[:, 2, 0])
    roll = torch.arctan(cam_R[:, 1, 0]/cam_R[:, 0, 0])
    return pitch, yaw, roll

def R_from_pitch_yaw_roll(pitch, yaw, roll, system=None):
    '''
    Retrieve the camera rotation from pitch, yaw, roll angles.
    Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the left, up,
    forward vector relative to the world system.
    Hence, R = R_z(roll)Ry_(yaw)Rx_(pitch)
    '''
    R = torch.eye(3, device=yaw.device).repeat(*yaw.shape, 1, 1)
    R[..., 0, 0] = torch.cos(yaw) * torch.cos(roll)
    R[..., 0, 1] = torch.sin(pitch) * torch.sin(yaw) * torch.cos(roll) - torch.cos(pitch) * torch.sin(roll)
    R[..., 0, 2] = torch.cos(pitch) * torch.sin(yaw) * torch.cos(roll) + torch.sin(pitch) * torch.sin(roll)
    R[..., 1, 0] = torch.cos(yaw) * torch.sin(roll)
    R[..., 1, 1] = torch.sin(pitch) * torch.sin(yaw) * torch.sin(roll) + torch.cos(pitch) * torch.cos(roll)
    R[..., 1, 2] = torch.cos(pitch) * torch.sin(yaw) * torch.sin(roll) - torch.sin(pitch) * torch.cos(roll)
    R[..., 2, 0] = - torch.sin(yaw)
    R[..., 2, 1] = torch.sin(pitch) * torch.cos(yaw)
    R[..., 2, 2] = torch.cos(pitch) * torch.cos(yaw)

    if system == 'opengl_cam':
        # transform to opengl cam_Rs
        R[..., 0] *= -1
        R[..., 2] *= -1

    return R


def R_from_yaw(yaw):
    '''
    Retrieve the rotation only from yaw angles, This is for box orientations.
    refer to utils.tools.R_from_pitch_yaw_roll
    Orientation setting. R:=[v1, v2, v3], the three column vectors respectively denote the left, up,
    forward vector relative to the world system.
    R = Ry_(yaw)
    '''
    R = torch.eye(3, device=yaw.device).repeat(*yaw.shape, 1, 1)
    R[..., 0, 0] = torch.cos(yaw)
    R[..., 0, 2] = torch.sin(yaw)
    R[..., 2, 0] = -torch.sin(yaw)
    R[..., 2, 2] = torch.cos(yaw)
    return R

def get_box_corners(centers, vectors, return_faces=False):
    '''
    Convert box center and vectors to the corner-form.
    Note x0<x1, y0<y1, z0<z1, then the 8 corners are concatenated by:
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
     [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]
    :return: corner points and faces related to the box
    '''
    corner_pnts = []

    corner_pnts.append(centers - vectors[:, :, 0] - vectors[:, :, 1] - vectors[:, :, 2])
    corner_pnts.append(centers - vectors[:, :, 0] - vectors[:, :, 1] + vectors[:, :, 2])
    corner_pnts.append(centers - vectors[:, :, 0] + vectors[:, :, 1] - vectors[:, :, 2])
    corner_pnts.append(centers - vectors[:, :, 0] + vectors[:, :, 1] + vectors[:, :, 2])

    corner_pnts.append(centers + vectors[:, :, 0] - vectors[:, :, 1] - vectors[:, :, 2])
    corner_pnts.append(centers + vectors[:, :, 0] - vectors[:, :, 1] + vectors[:, :, 2])
    corner_pnts.append(centers + vectors[:, :, 0] + vectors[:, :, 1] - vectors[:, :, 2])
    corner_pnts.append(centers + vectors[:, :, 0] + vectors[:, :, 1] + vectors[:, :, 2])

    corner_pnts = torch.stack(corner_pnts, dim=2)

    if return_faces:
        faces = [(0, 1, 3, 2), (1, 5, 7, 3), (4, 6, 7, 5), (0, 2, 6, 4), (0, 4, 5, 1), (2, 3, 7, 6)]
        return corner_pnts, faces
    else:
        return corner_pnts

def project_points_to_2d(points, cam_Ks, cam_Ts, eps=1e-3):
    '''
    transform box corners to cam system
    :param points: N x 3 coordinates in world system
    :param cam_K: cam K matrix
    :param cam_T: 4x4 extrinsic matrix with open-gl setting. (http://www.songho.ca/opengl/gl_camera.html)
                  [[v1, v2, v3, T]
                   [0,  0,  0,  1,]]
                  where v1, v2, v3 corresponds to right, up, backward of a camera
    :param eps: for points on the back side of camera.
    '''
    '''transform to camera system'''
    n_cams = cam_Ts.size(1)
    n_objects = points.size(1)
    device = points.device
    # reorganize points to n_batch x n_cam x n_object x n_corner x xyz
    points = points.unsqueeze(1).repeat(1, n_cams, 1, 1, 1)
    # reorganize cam_Ts to n_batch x n_cam x n_object x 4 x 4
    cam_Ts = cam_Ts.unsqueeze(2).repeat(1, 1, n_objects, 1, 1)
    # reorganize cam_Ts to n_batch x n_cam x n_object x 3 x 3
    cam_Ks = cam_Ks.unsqueeze(2).repeat(1, 1, n_objects, 1, 1)

    # homogeneous coordinates
    points_h = torch.cat([points, torch.ones(size=(*points.shape[:-1], 1), device=device)], dim=-1)
    points_cam = torch.einsum('bijmn,bijnq->bijmq', torch.linalg.inv(cam_Ts), points_h.transpose(-1, -2))
    points_cam = points_cam[:, :, :, :3]

    # transform to opencv system
    points_cam[:, :, :, 1] *= -1
    points_cam[:, :, :, 2] *= -1

    # project to image plane
    in_frustum = (points_cam[:, :, :, 2] > eps)
    points_cam_h = points_cam / torch.clamp(points_cam[:, :, :, [2]], min=eps)
    pixels = torch.einsum('bijmn,bijnq->bijmq', cam_Ks[:, :, :, :2, :2], points_cam_h[:, :, :, :2])
    pixels = cam_Ks[:, :, :, :2, [2]] + pixels
    pixels = pixels.transpose(-1, -2)

    return pixels, in_frustum

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((*true_labels.shape[:2], classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(2, true_labels.data.unsqueeze(2), confidence)
    return true_dist

def process_box2ds(x1y1x2y2, cls_scores, image_sizes):
    box2d_centers = (x1y1x2y2[..., :2] + x1y1x2y2[..., 2:4]) / 2
    box2d_sizes = (x1y1x2y2[..., 2:4] - x1y1x2y2[..., :2]) + 1

    box2d_centers = torch.div(box2d_centers, image_sizes - 1)
    box2d_sizes = torch.div(box2d_sizes, image_sizes)
    box2ds = torch.cat([box2d_centers, box2d_sizes, cls_scores], dim=-1)

    return box2ds


def get_box2ds_from_net(box2ds, type, **kwargs):
    '''
    Process box2ds from network and feed them to discriminator.
    :return: the output box2ds will be in the form of [2d center, 2d size, n_d class]
    '''
    if type == 'pred':
        # Get pred box2ds.
        return box2ds.transpose(1, 2)
    elif type == 'gt':
        image_sizes = box2ds['image_size']
        # Get gt box2ds.
        gt_x1y1x2y2 = torch.cat([box2ds['box2ds'][..., :2], box2ds['box2ds'][..., :2] + box2ds['box2ds'][..., 2:4] - 1],
                                dim=-1)
        category_ids = smooth_one_hot(box2ds['category_ids'], classes=kwargs['n_classes'], smoothing=0.1)
        # category_ids = F.gumbel_softmax(torch.log(category_ids), dim=-1)
        # F.one_hot(gt['category_ids'], num_classes=self.n_classes)
        gt_box2ds = process_box2ds(gt_x1y1x2y2, category_ids, image_sizes[:, None])
        gt_box2ds = gt_box2ds.transpose(1, 2)
        return gt_box2ds
    else:
        raise NameError('Wrong box2ds type.')

def cartesian_prod(*xs):
    ns = len(xs)
    return torch.stack(torch.meshgrid(*xs, indexing='ij'), dim=-1).view(-1, ns)

def cwcywh2x1y1x2y2(cwcywh):
    '''Transform the cwcywh box into x1y1x2y2 box.'''
    cwcy = cwcywh[..., :2]
    wh = cwcywh[..., 2:4]
    x1y1x2y2 = torch.cat([cwcy - (wh - 1) / 2, cwcy + (wh - 1) / 2], dim=-1)
    return x1y1x2y2

def normalize_x1y1x2y2(x1y1x2y2, image_size):
    x1y1x2y2 = torch.div(x1y1x2y2.view(*x1y1x2y2.shape[:-1], 2, 2), (image_size[:, None, :, None] - 1))
    x1y1x2y2 = x1y1x2y2.flatten(-2, -1)
    return x1y1x2y2
