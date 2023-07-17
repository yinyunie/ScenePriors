#  Copyright (c) 8.2022. Yinyu Nie
#  License: MIT
import random
import numpy as np
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
import argparse
from pathlib import Path
import h5py
import cv2
from torch import nn
from models.ours.modules.hidden_to_output import DeterminsticOutput
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
from net_utils.box_utils import get_box_corners, project_points_to_2d, normalize_x1y1x2y2
from external.fast_transformers.fast_transformers.masking import LengthMask
from torch.nn import functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    MeshRasterizer,
    MeshRendererWithFragments)
from torch.nn import L1Loss, CrossEntropyLoss, CosineSimilarity, BCELoss
from net_utils.matcher_tracking import HungarianMatcher, generalized_box_iou
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="Overfit a single ScanNet scene.")
    parser.add_argument("--data_path", type=str,
                        default='/home/ynie/Projects/SceneSynthesis/datasets/ScanNet/ScanNet_samples/Apartment',
                        help="Give the absolute path of all view data.")
    parser.add_argument("--room_uid", type=str,
                        default='scene0000_00',
                        help="Give the room uid you would like to reconstruct.")
    parser.add_argument("--downsample_ratio", type=int,
                        default=4,
                        help="Downsample rendering image to boost training speed.")
    return parser.parse_args()

default_collate = torch.utils.data.dataloader.default_collate
label_names = [
    'void',
    'bathtub', 'bed', 'bookshelf', 'cabinet', 'chair',
    'counter', 'desk', 'dresser', 'lamp', 'night stand',
    'refridgerator', 'shelves', 'sink', 'sofa', 'table',
    'television', 'toilet', 'whiteboard']
n_classes = len(label_names)
render_image_size = np.array((1296, 968))


def parse_hdf5(sample_file):
    '''read data'''
    with h5py.File(sample_file, "r") as sample_data:
        # img = Image.fromarray(sample_data['colors'][:])
        # img = self.preprocess(img)
        img = None
        cam_T = sample_data['cam_T'][:]
        cam_K = sample_data['cam_K'][:]
        image_size = sample_data['image_size'][:]
        inst_h5py = sample_data['inst_info']
        box2ds = []
        category_ids = []
        inst_marks = []
        masks = []
        for inst_id in inst_h5py:
            box2ds.append(inst_h5py[inst_id]['bbox2d'][:])
            category_ids.append(inst_h5py[inst_id]['category_id'][0])
            inst_marks.append(inst_h5py[inst_id]['inst_mark'][0])
            masks.append(inst_h5py[inst_id]['mask'][:])

    insts = {'box2ds': box2ds,
             'category_ids': category_ids,
             'inst_marks': inst_marks,
             'masks': masks}
    return img, cam_K, cam_T, insts, image_size


def track_insts(parsed_data, unique_marks):

    box2ds_template = -1 * np.ones(shape=(len(unique_marks), 4), dtype=np.int32)
    category_ids_template = np.zeros(shape=(len(unique_marks),), dtype=np.int32)
    masks_template = np.array([None] * len(unique_marks))
    inst_marks_template = np.zeros(shape=(len(unique_marks),), dtype=bool)
    for view_data in parsed_data:
        insts_data = view_data[3]
        ordering = [unique_marks.index(mark) for mark in insts_data['inst_marks']]
        # re-order instances
        # bbox2ds
        empty_box2ds = box2ds_template.copy()
        empty_box2ds[ordering, :] = insts_data['box2ds']
        insts_data['box2ds'] = empty_box2ds
        # category_ids
        empty_category_ids = category_ids_template.copy()
        empty_category_ids[ordering] = insts_data['category_ids']
        insts_data['category_ids'] = empty_category_ids
        # masks
        empty_masks = masks_template.copy()
        empty_masks[ordering] = insts_data['masks']
        insts_data['masks'] = empty_masks
        # inst_marks
        empty_inst_marks = inst_marks_template.copy()
        empty_inst_marks[ordering] = True
        insts_data['inst_marks'] = empty_inst_marks

    return parsed_data


def read_data(args):
    view_files = Path(args.data_path).rglob(args.room_uid + '_*')
    view_files = [file for idx, file in enumerate(view_files)]

    # augment data
    theta = np.random.choice([0, 0.5*np.pi, np.pi, 1.5*np.pi], 1)[0]
    rot_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    offset = 2 * np.random.random(3) - 1
    offset[1] = 0
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = offset

    parsed_data = []
    for view_file in view_files:
        img, cam_K, cam_T, insts, image_size = parse_hdf5(view_file)
        cam_T = trans_mat.dot(cam_T)
        parsed_data.append((img, cam_K, cam_T, insts, image_size))

    '''organize objects following unique instance ids.'''
    # all unique instance marks in this scene
    unique_marks = list(set([mark for item in parsed_data for mark in item[3]['inst_marks']]))

    # re-organize instances following track ids
    parsed_data = track_insts(parsed_data, unique_marks)

    views_data = []
    for (img, cam_K, cam_T, insts, image_size) in parsed_data:
        inst_marks = insts['inst_marks']
        n_objects = len(insts['box2ds'])
        box2ds = np.array(insts['box2ds'])
        category_ids = insts['category_ids']
        category_labels = np.zeros(shape=(n_objects, n_classes))
        category_labels[range(n_objects), category_ids] = 1
        x1y1 = box2ds[..., :2]
        x2y2 = box2ds[..., :2] + box2ds[..., 2:4] - 1
        # box2d_centers = box2d_centers / (image_sizes - 1)
        # box2d_sizes = box2d_sizes / image_sizes
        inst_box2ds = np.concatenate([x1y1, x2y2, category_labels], axis=-1)
        inst_masks = -1 * np.ones((int(image_size[1]), int(image_size[0])), dtype=int)
        render_mask = np.ones_like(inst_masks)

        for inst_id, (box2d, mask) in enumerate(zip(insts['box2ds'], insts['masks'])):
            if not inst_marks[inst_id]:
                continue
            current_block = inst_masks[box2d[1]: box2d[1] + box2d[3], box2d[0]: box2d[0] + box2d[2]]
            current_block[mask == True] = inst_id

        if (render_image_size != image_size).any():
            scale_ratio = render_image_size / image_size
            to_size = np.int32(image_size * scale_ratio.min())
            inst_masks = cv2.resize(inst_masks, to_size, interpolation=cv2.INTER_NEAREST_EXACT)
            render_mask = np.ones_like(inst_masks)
            long_axis = np.argmax(scale_ratio)
            padding = render_image_size[long_axis] - image_size[long_axis]
            if long_axis == 0:
                pad_item = ((0, 0), (padding // 2, padding - padding // 2))
            else:
                pad_item = ((padding // 2, padding - padding // 2), (0, 0))
            inst_masks = np.pad(inst_masks, pad_item, 'constant', constant_values=((-1, -1), (-1, -1)))
            render_mask = np.pad(render_mask, pad_item, 'constant', constant_values=((0, 0), (0, 0)))

        resize_w = int(image_size[0]) // args.downsample_ratio
        resize_h = int(image_size[1]) // args.downsample_ratio
        inst_masks = cv2.resize(inst_masks, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST_EXACT)
        render_mask = cv2.resize(render_mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST_EXACT)

        '''store gt data'''
        data = {}
        # data['img'] = img
        data['cam_K'] = cam_K.astype(np.float32)
        data['image_size'] = image_size.astype(np.float32)
        data['cam_T'] = cam_T.astype(np.float32)
        data['box2ds_tr'] = inst_box2ds.astype(np.float32)
        data['max_len'] = len(unique_marks)
        data['masks_tr'] = inst_masks.astype(np.int64)
        data['render_mask_tr'] = render_mask.astype(bool)
        data['inst_marks'] = np.array(inst_marks, dtype=bool)
        views_data.append(data)

    return views_data

def collate_fn(samples):
    padding_keys = ['box2ds_tr']
    room_level_keys = ['inst_marks']
    max_length = max(sample[0]['max_len'] for sample in samples)

    collated_batch = {}
    for key in samples[0][0]:
        collated_batch[key] = []
        for sample in samples:  # n_batch samples
            if key in room_level_keys:
                collated_view_data = default_collate([np.append(view_data[key], np.zeros(
                    (max_length - len(view_data[key]), *view_data[key].shape[1:]), dtype=view_data[key].dtype)) for
                                                      view_data in sample])
            elif key not in padding_keys:
                collated_view_data = default_collate([view_data[key] for view_data in sample])
            else:
                collated_view_data = default_collate([np.vstack([view_data[key], np.zeros(
                    (max_length - len(view_data[key]), *view_data[key].shape[1:]), dtype=view_data[key].dtype)]) for view_data in
                                                      sample])
            collated_batch[key].append(collated_view_data)

    for key in collated_batch:
        if key not in ['sample_name']:
            collated_batch[key] = default_collate(collated_batch[key])

    return collated_batch

def to_device(data, device='cuda'):
    for key in data:
        if key in ['sample_name']: continue
        data[key] = data[key].to(device)
    return data

class LatentEncoder(nn.Module):
    def __init__(self, max_len):
        super(LatentEncoder, self).__init__()
        shared_feat = 128
        latent_codes = torch.randn(1, max_len, shared_feat)
        self.latent_codes = nn.Parameter(latent_codes, requires_grad=True)
        bbox_feat = 128
        shape_feat = 128
        self.mlp_bbox = nn.Sequential(nn.Linear(shared_feat, 128), nn.ReLU(),
                                      nn.Linear(128, bbox_feat))
        self.shape_bbox = nn.Sequential(nn.Linear(shared_feat, 128), nn.ReLU(),
                                        nn.Linear(128, shape_feat))

    def forward(self):
        latent_codes = self.latent_codes
        box_feat = self.mlp_bbox(latent_codes)
        shape_feat = self.shape_bbox(latent_codes)
        return {'box_feat': box_feat,
                'shape_feat': shape_feat}

class BoxDecoder(nn.Module):
    def __init__(self):
        super(BoxDecoder, self).__init__()
        d_model = 128
        self.hidden2output = DeterminsticOutput(hidden_size=d_model,
                                                n_classes=n_classes,
                                                with_extra_fc=False)
    def forward(self, box_feat):
        box3ds = self.hidden2output(box_feat)
        return box3ds

class ShapeDecoder(nn.Module):
    def __init__(self):
        super(ShapeDecoder, self).__init__()
        self.src_mesh = ico_sphere(4, 'cuda')
        latent_dim = 128
        self.mlp_shape = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                       nn.Linear(128, 128))
        self.mlp_deform = nn.Sequential(nn.Linear(latent_dim + 3, 128), nn.ReLU(),
                                        nn.Linear(128, 64), nn.ReLU(),
                                        nn.Linear(64, 3))

    def forward(self, shape_feat, start_render):
        shape_feat = self.mlp_shape(shape_feat)
        n_batch, n_object, feat_dim = shape_feat.shape
        shape_feat = shape_feat.view(n_batch * n_object, feat_dim)
        meshes = self.src_mesh.extend(n_batch * n_object)
        vertices = meshes.verts_padded()
        n_vertices = vertices.size(1)
        shape_feat = shape_feat[:, None].expand(-1, n_vertices, -1)
        shape_feat = torch.cat([shape_feat, vertices], dim=-1)
        offsets = self.mlp_deform(shape_feat)
        offsets = offsets.view(-1, 3) * start_render
        meshes = meshes.offset_verts(offsets)
        return meshes

class DiffRender(nn.Module):
    def __init__(self, device='cuda', n_views=10, downsample_ratio=4):
        super(DiffRender, self).__init__()
        # set rasterizer
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=(int(render_image_size[1])//downsample_ratio, int(render_image_size[0])//downsample_ratio),
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50
        )
        rasterizer = MeshRasterizer(
            raster_settings=raster_settings
        )
        shader = SoftSilhouetteShader()

        # initialize cameras
        self.cameras = PerspectiveCameras(
            focal_length=torch.zeros(n_views, 2, device=device),
            principal_point=torch.zeros(n_views, 2, device=device),
            image_size=torch.ones(n_views, 2, device=device),
            R=torch.zeros(n_views, 3, 3, device=device),
            T=torch.zeros(n_views, 3, device=device),
            in_ndc=False,
            device=device)
        self.renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=shader)
        self.vertices_per_template = 2562
        self.faces_per_template = 5120
        self.device = device

    def project_points_to_2d(self, points, cam_Ks, cam_Ts, image_sizes):
        n_batch, n_view = cam_Ts.shape[:2]
        n_objects = points.size(1)

        fcl_screen = torch.cat([cam_Ks[:, :, 0, [0]], cam_Ks[:, :, 1, [1]]], dim=-1) # n_scene x n_view x 2
        prp_screen = cam_Ks[..., :2, 2]  # n_scene x n_view x 2
        image_sizes = image_sizes[..., [1, 0]]
        '''transform to camera system'''
        # reorganize points to n_batch x n_cam x n_object x n_corner x xyz
        points = points.unsqueeze(1).repeat(1, n_view, 1, 1, 1)
        # reorganize cam_Ts to n_batch x n_cam x n_object x 4 x 4
        cam_Ts = cam_Ts.unsqueeze(2).repeat(1, 1, n_objects, 1, 1)
        # reorganize cam_Ts to n_batch x n_cam x n_object x 3 x 3
        # cam_Ks = cam_Ks.unsqueeze(2).repeat(1, 1, n_objects, 1, 1)
        fcl_screen = fcl_screen.unsqueeze(2).repeat(1, 1, n_objects, 1)
        prp_screen = prp_screen.unsqueeze(2).repeat(1, 1, n_objects, 1)
        image_sizes = image_sizes.unsqueeze(2).repeat(1, 1, n_objects, 1)

        points = points.flatten(0, 2)
        cam_Ts = cam_Ts.flatten(0, 2)
        # cam_Ks = cam_Ks.flatten(0, 2)
        fcl_screen = fcl_screen.flatten(0, 2)
        prp_screen = prp_screen.flatten(0, 2)
        image_sizes = image_sizes.flatten(0, 2)

        Rs = cam_Ts[:, :3, :3]
        # transform to pytorch3d camera system
        Rs[..., 0] *= -1
        Rs[..., 2] *= -1 # n_scene * n_view x 3 x 3

        cam_loc = cam_Ts[:, :3, 3]
        Ts = -torch.bmm(Rs.transpose(1, 2), cam_loc[:, :, None])[:, :, 0] # n_scene * n_view x 3

        # x -> width direction
        # y -> height direction
        points2d = self.cameras.transform_points_screen(
            points,
            focal_length=fcl_screen,
            principal_point=prp_screen,
            image_size=image_sizes,
            R=Rs,
            T=Ts,
            eps=1e-3)[..., :2]

        # transform = cameras.get_full_projection_transform()
        # points2d = transform.transform_points(points, eps=1e-3)[..., :2]
        # points2d = 2 * cameras.get_principal_point()[:, None] - points2d

        points2d = points2d.view(n_batch, n_view, n_objects, *points2d.shape[1:])

        return points2d

    @staticmethod
    def render_bbox(centers, sizes, cam_Ts, cam_Ks, image_sizes):
        # get box and project
        vectors = torch.diag_embed(sizes / 2.)
        box_corners = get_box_corners(centers, vectors)
        # get projected 2d bboxes
        # n_batch x n_views x n_objects x 8 x 2
        # proj_box2ds = self.project_points_to_2d(box_corners, cam_Ks, cam_Ts, image_sizes)
        proj_box2ds, in_frustum = project_points_to_2d(box_corners, cam_Ks, cam_Ts)

        in_frustum = in_frustum.max(dim=-1)[0]

        # clamp to image sizes
        x1y1x2y2 = torch.clamp(torch.min(proj_box2ds, image_sizes[:, :, None, None] - 1), min=0)
        x1y1x2y2 = torch.cat([torch.min(x1y1x2y2, dim=3)[0], torch.max(x1y1x2y2, dim=3)[0]], dim=-1)

        # # draw predicted box2ds
        # from PIL import Image, ImageDraw
        # image = np.zeros(shape=(360, 480, 3), dtype=np.uint8)
        # image = Image.fromarray(image).convert("RGB")
        # img_draw = ImageDraw.Draw(image)
        # view_id = 0
        # x1y1x2y2_view = x1y1x2y2[0, view_id].detach().cpu().numpy()
        # for per_x1y1x2y2 in x1y1x2y2_view:
        #     img_draw.rectangle(per_x1y1x2y2, width=3)
        # image.show()

        # box2d_centers = (x1y1x2y2[..., :2] + x1y1x2y2[..., 2:4]) / 2
        # box2d_sizes = (x1y1x2y2[..., 2:4] - x1y1x2y2[..., :2]) + 1
        # box2d_centers = torch.div(box2d_centers, image_sizes[:, :, None] - 1)
        # box2d_sizes = torch.div(box2d_sizes, image_sizes[:, :, None])

        return x1y1x2y2, in_frustum

    def move_meshes_to_box3ds(self, meshes, centers, sizes):
        vertices = meshes.verts_padded()

        centers = centers.flatten(0, 1)[:, None]
        centers = centers.expand(-1, self.vertices_per_template, -1).contiguous()

        sizes = sizes.flatten(0, 1)[:, None]
        sizes = sizes.expand(-1, self.vertices_per_template, -1).contiguous()
        sizes = sizes / 2.

        vertices = vertices * sizes + centers

        meshes = meshes.update_padded(new_verts_padded=vertices)
        return meshes

    def render_instances(self, meshes, cam_Ts, cam_Ks, image_sizes, render_mask_tr):
        n_batch, n_view = cam_Ts.shape[:2]
        image_sizes = image_sizes[..., [1, 0]]
        n_object = len(meshes) // n_batch
        # merge meshes into scenes by batch
        scene_splits = torch.arange(n_batch * n_object).split(n_object)
        # n_scene x n_object
        scenes = [join_meshes_as_scene([meshes[idx.item()] for idx in scene_split]) for scene_split in scene_splits]
        # n_scene x n_view x n_object
        scenes_ext = join_meshes_as_batch([scene.extend(n_view) for scene in scenes])

        # set cameras (n_scene x n_view x dim)
        fcl_screen = torch.cat([cam_Ks[:, :, 0, [0]], cam_Ks[:, :, 1, [1]]], dim=-1).view(n_batch * n_view, 2)  # n_scene * n_view x 2
        prp_screen = cam_Ks[..., :2, 2].view(n_batch * n_view, 2)  # n_scene * n_view x 2
        image_sizes = image_sizes.view(n_batch * n_view, 2)  # n_scene * n_view x 2
        cam_Ts = cam_Ts.view(n_batch * n_view, 4, 4)  # n_scene * n_view x 4 x 4

        Rs = cam_Ts[:, :3, :3]
        # transform to pytorch3d camera system
        Rs[..., 0] *= -1
        Rs[..., 2] *= -1  # n_scene * n_view x 3 x 3

        cam_loc = cam_Ts[:, :3, 3]
        Ts = -torch.bmm(Rs.transpose(1, 2), cam_loc[:, :, None])[:, :, 0]  # n_scene * n_view x 3

        silhouettes, fragments = self.renderer(scenes_ext,
                                               cameras=self.cameras,
                                               focal_length=fcl_screen,
                                               principal_point=prp_screen,
                                               image_size=image_sizes,
                                               R=Rs,
                                               T=Ts,
                                               eps=1e-3)
        silhouettes = silhouettes[..., 3]
        silhouettes = silhouettes.view(n_batch, n_view, *silhouettes.shape[1:])
        # silhouettes = silhouettes * render_mask_tr

        '''Get instance masks'''
        instance_labels = fragments.pix_to_face[..., 0]

        render_mask_tr = torch.logical_not(render_mask_tr.flatten(0, 1))
        instance_labels = instance_labels.masked_fill(render_mask_tr, -1)

        # scene_ids = torch.div(instance_labels, (n_view * n_object * self.faces_per_template), rounding_mode='floor')
        remaining = torch.remainder(instance_labels, (n_view * n_object * self.faces_per_template))
        # view_ids = torch.div(remaining, (n_object * self.faces_per_template), rounding_mode='floor')
        remaining = torch.remainder(remaining, (n_object * self.faces_per_template))
        obj_ids = torch.div(remaining, (self.faces_per_template), rounding_mode='floor')
        # face_ids = torch.remainder(remaining, (self.faces_per_template))

        obj_ids[instance_labels < 0] = -1
        obj_ids = obj_ids.view(n_batch, n_view, *obj_ids.shape[1:])

        return silhouettes, obj_ids

    @staticmethod
    def project_points(meshes, cam_Ts, cam_Ks, image_sizes):
        n_batch, n_view = cam_Ts.shape[:2]
        n_object = len(meshes) // n_batch

        # get 3d points and project
        points_on_meshes = meshes.verts_padded()
        points_on_meshes = points_on_meshes.view(n_batch, n_object, -1, 3)

        proj_points, in_frustum = project_points_to_2d(points_on_meshes, cam_Ks, cam_Ts)

        in_frustum = in_frustum.max(dim=-1)[0]

        # clamp to image sizes
        proj_points = torch.clamp(torch.min(proj_points, image_sizes[:, :, None, None] - 1), min=0)

        return proj_points, in_frustum, points_on_meshes

    def forward(self, box3ds, meshes, cam_Ts, cam_Ks, image_sizes, render_mask_tr, start_render=True):
        centers = box3ds[..., :3]
        sizes = box3ds[..., 3:6]
        classes = box3ds[..., 6:]

        # move meshes to box3ds
        meshes = self.move_meshes_to_box3ds(meshes, centers, sizes)

        # render points on meshes to points on 2D
        points_2d, in_frustum, points_on_meshes = self.project_points(meshes, cam_Ts.clone(), cam_Ks.clone(), image_sizes.clone())

        '''render meshes to silhouettes'''
        if start_render:
            # from pytorch3d.vis.plotly_vis import plot_batch_individually, plot_scene
            # fig = plot_scene({
            #     "subplot1": {"mesh%d"%(i):meshes[i] for i in range(len(meshes))}
            # })
            # fig.show()

            silhouettes, obj_ids = self.render_instances(meshes, cam_Ts.clone(), cam_Ks.clone(), image_sizes.clone(),
                                                         render_mask_tr)
        else:
            silhouettes = None
            obj_ids = None

        return {'points_2d': points_2d,
                'points_3d': points_on_meshes,
                'in_frustum': in_frustum,
                'classes': classes,
                'silhouettes': silhouettes,
                'obj_ids': obj_ids}

class MultiViewRenderLoss(object):
    def __init__(self, weight=1, device='cuda'):
        self.weight = weight
        self.device = device
        self.l1_loss = L1Loss(reduction='none')
        self.cos_sim = CosineSimilarity(dim=-1)
        self.matcher = HungarianMatcher(1, 5)
        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.bce_loss = BCELoss(reduction='none')

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

    def views_loss(self, est_data, gt_data, start_deform=False):
        '''Calculate rendering loss.'''
        # indicates the instance marks for each object
        gt_obj_view_mask = gt_data['inst_marks']
        # indicates how many objects occur in each scene
        obj_lens = gt_data['max_len'][:, 0]
        # indicates which est objects are used for loss calculation
        pred_mask = LengthMask(obj_lens).bool_matrix

        '''prepare est data'''
        est_points_2d = est_data['points_2d']
        est_cls_scores = est_data['classes']
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
            return frustum_loss, box_cls_loss, box_loss, mask_loss

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

        return frustum_loss, box_cls_loss, box_loss, mask_loss

    def __call__(self, est_data, gt_data, start_render):
        '''Calculate rendering loss'''
        frustum_loss, box_cls_loss, box_loss, mask_loss = self.views_loss(est_data, gt_data, start_render)
        total_loss = frustum_loss + box_cls_loss + 5 * box_loss

        if start_render:
            total_loss = total_loss + 2 * mask_loss

        return {'total': total_loss * self.weight,
                'frustum_loss': frustum_loss,
                'cls_loss': box_cls_loss, 'box_loss': box_loss,
                'mask_loss': mask_loss}


def get_optimizer(modules, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    optim_params = list()
    for module in modules:
        optim_params.append(
            {'params': filter(lambda p: p.requires_grad, module['name'].parameters()),
             'lr': float(module['lr']),
             'betas': tuple(betas),
             'eps': float(eps),
             'weight_decay': float(weight_decay)}
        )


    optimizer = torch.optim.AdamW(optim_params,
                                  lr=float(lr),
                                  betas=tuple(betas),
                                  eps=float(eps),
                                  weight_decay=float(weight_decay))
    return optimizer


def get_sample(all_samples, n_views):
    selected_view_ids = np.random.choice(all_samples['cam_K'].shape[1], n_views, replace=False)
    sample = {}
    for key, item in all_samples.items():
        sample[key] = item[:, selected_view_ids]
    return sample


if __name__ == '__main__':
    args = parse_args()
    '''Load data'''
    views_data = read_data(args)
    all_samples = collate_fn([views_data])
    all_samples = to_device(all_samples)

    n_view_batch = 10
    '''Define latent code for each object'''
    max_len = all_samples['max_len'].max()
    latent_encoder = LatentEncoder(max_len=max_len).to('cuda')
    box_decoder = BoxDecoder().to('cuda')
    shape_decoder = ShapeDecoder().to('cuda')
    renderer = DiffRender(n_views=n_view_batch, downsample_ratio=args.downsample_ratio).to('cuda')
    rendering_loss = MultiViewRenderLoss()

    '''Define training strategy'''
    epochs = 2000
    lr = 0.001
    modules = [{'name': latent_encoder, 'lr': lr},
               {'name': box_decoder, 'lr': lr},
               {'name': shape_decoder, 'lr': lr},
               {'name': renderer, 'lr': lr}]
    optimizer = get_optimizer(modules)

    '''Training'''
    latent_encoder.train(True)
    box_decoder.train(True)
    shape_decoder.train(True)
    renderer.train(True)
    start = time()

    for epoch in range(epochs):
        start_render = (epoch>=500)
        sample = get_sample(all_samples, n_view_batch)
        optimizer.zero_grad()
        latent_codes = latent_encoder()
        box3ds = box_decoder(latent_codes['box_feat'])
        shapes = shape_decoder(latent_codes['shape_feat'], start_render)
        renderings = renderer(box3ds, shapes, sample['cam_T'], sample['cam_K'], sample['image_size'],
                              sample['render_mask_tr'], start_render)
        loss = rendering_loss(renderings, sample, start_render=start_render)

        loss['total'].backward()
        optimizer.step()

        print('=' * 100)
        print(epoch)
        for key, item in loss.items():
            print(key, ':', item.item())

    print('Downsample ratio %d:Time elapsed: %f' % (args.downsample_ratio, time()-start))