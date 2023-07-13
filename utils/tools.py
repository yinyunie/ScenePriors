#  Copyright (c) 1.2022. Yinyu Nie
#  License: MIT
import json
import csv
from typing import Union, Dict, List
import numpy as np
import os
from skimage import measure
import h5py
from shapely.geometry.polygon import Polygon


def read_mapping_csv(file, from_label, to_label):
    with open(file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        mapping_dict = dict()
        for row in reader:
            mapping_dict[row[from_label]] = row[to_label]
        return mapping_dict

def label_mapping_2D(img: np.ndarray, mapping_dict: Dict):
    '''To map the labels in img following the rule in mapping_dict.'''
    out_img = np.zeros_like(img)
    existing_labels = np.unique(img)
    for label in existing_labels:
        out_img[img==label] = mapping_dict[label]
    return out_img

def read_json(file):
    '''
    read json file
    @param file: file path.
    @return:
    '''
    with open(file, 'r') as f:
        output = json.load(f)
    return output


def write_json(file, data):
    '''
    read json file
    @param file: file path.
    @param data: dict content
    @return:
    '''
    assert os.path.exists(os.path.dirname(file))

    with open(file, 'w') as f:
        json.dump(data, f)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(np.array(contours, dtype=object), 1)
    for contour in contours:
        contour = contour.astype(float)
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def normalize(a, axis=-1, order=2):
    '''
    Normalize any kinds of tensor data along a specific axis
    :param a: source tensor data.
    :param axis: data on this axis will be normalized.
    :param order: Norm order, L0, L1 or L2.
    :return:
    '''
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    if len(a.shape) == 1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)

def heading2arc(vec: Union[np.ndarray, List]):
    '''
    transform orientation vector to arc angles (to z-axis).
    :param vec: Nx3 nd array.
    :return:
    '''
    if isinstance(vec, list):
        vec = np.array(vec)
    if len(vec.shape) == 1:
        vec = vec[np.newaxis]
    return np.arctan2(vec[:, 0], vec[:, 2])

def arc2heading(arc):
    '''transform arc angles (to z-axis) to orientation vector'''
    if isinstance(arc, (float, int)):
        arc = np.array([arc])
    heading = np.zeros(shape=(len(arc), 3))
    heading[:, 0] = np.sin(arc)
    heading[:, 2] = np.cos(arc)
    return heading

def pitch_yaw_roll_from_R(cam_R):
    '''
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
        cam_R = cam_R[np.newaxis]
    # in normal cases (capturing in a scene)
    # pitch ranges from [-pi/2, pi/2]
    # yaw ranges from [-pi, pi]
    # roll ranges from [-pi/2, pi/2]
    pitch = np.arctan(cam_R[:, 2, 1]/cam_R[:, 2, 2])
    yaw = np.arcsin(-cam_R[:, 2, 0])
    roll = np.arctan(cam_R[:, 1, 0]/cam_R[:, 0, 0])
    return pitch, yaw, roll

def R_from_pitch_yaw_roll(pitch, yaw, roll):
    '''
    Retrieve the camera rotation from pitch, yaw, roll angles.
    Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the left, up,
    forward vector relative to the world system.
    Hence, R = R_z(roll)Ry_(yaw)Rx_(pitch)
    '''
    if isinstance(pitch, (float, int)):
        pitch = np.array([pitch])
    if isinstance(yaw, (float, int)):
        yaw = np.array([yaw])
    if isinstance(roll, (float, int)):
        roll = np.array([roll])
    R = np.zeros((len(pitch), 3, 3))
    R[:, 0, 0] = np.cos(yaw) * np.cos(roll)
    R[:, 0, 1] = np.sin(pitch) * np.sin(yaw) * np.cos(roll) - np.cos(pitch) * np.sin(roll)
    R[:, 0, 2] = np.cos(pitch) * np.sin(yaw) * np.cos(roll) + np.sin(pitch) * np.sin(roll)
    R[:, 1, 0] = np.cos(yaw) * np.sin(roll)
    R[:, 1, 1] = np.sin(pitch) * np.sin(yaw) * np.sin(roll) + np.cos(pitch) * np.cos(roll)
    R[:, 1, 2] = np.cos(pitch) * np.sin(yaw) * np.sin(roll) - np.sin(pitch) * np.cos(roll)
    R[:, 2, 0] = - np.sin(yaw)
    R[:, 2, 1] = np.sin(pitch) * np.cos(yaw)
    R[:, 2, 2] = np.cos(pitch) * np.cos(yaw)
    return R

def sample_points_in_box(box, step_len=1, padding=0):
    '''
    Sample points in a node bounding box
    :param box:
        box['centroid']: 3D coordinates of box center
        box['R_mat']: [v1, v2, v3], respectively denote the vectors of left, up and forward.
        box['size']; edge length along v1, v2, v3
    :param step_len:
    :param padding:
    :return:
    '''
    centroid = box['centroid']
    size = box['size'] + padding
    R_mat = box['R_mat']

    '''sample points'''
    vectors = np.diag(size / 2.).dot(R_mat.T)
    bbox_corner = centroid - vectors[0] - vectors[1] - vectors[2]

    cx, cy, cz = np.meshgrid(np.arange(step_len, size[0], step_len),
                             np.arange(step_len, size[1], step_len),
                             np.arange(step_len, size[2], step_len), indexing='ij')
    cxyz = np.array([cx, cy, cz]).reshape(3, -1).T
    cxyz = cxyz[:, np.newaxis]
    R_mat = np.tile(R_mat, (cxyz.shape[0], 1, 1))
    points_in_cube = np.matmul(cxyz, R_mat) + bbox_corner
    return points_in_cube

def check_in_box(points, box_prop):
    '''Check if a point located in a box.
    R_mat: [v1, v2, v3], respectively denote the vectors of left, up and forward.'''
    centroid = np.array(box_prop['centroid'])
    size = np.array(box_prop['size'])
    R_mat = np.array(box_prop['R_mat'])

    offsets = points - centroid
    offsets_proj = np.abs(offsets.dot(R_mat))

    return np.min(offsets_proj <= size/2., axis=-1)

def filter_cam_locs(cam_locs, bbox_3ds):
    '''
    filter out the cam locs that are in nodes' bboxes
    :return: cam_loc ids that do not located in any bbox.
    '''
    inbox_vec = np.zeros(shape=(cam_locs.shape[:-1]), dtype=np.bool)
    for inst_bbox in bbox_3ds:
        centroid = inst_bbox[0:3]
        R_mat = R_from_pitch_yaw_roll(0, inst_bbox[6], 0)[0]
        size = inst_bbox[3:6]
        inbox_vec += check_in_box(cam_locs, {'centroid': centroid, 'size': size, 'R_mat': R_mat})
    return ~inbox_vec

def get_box_corners(center, vectors, return_faces=False):
    '''
    Convert box center and vectors to the corner-form.
    Note x0<x1, y0<y1, z0<z1, then the 8 corners are concatenated by:
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
     [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]
    :return: corner points and faces related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[2] = tuple(center - vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    corner_pnts[4] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[7] = tuple(center + vectors[0] + vectors[1] + vectors[2])

    if return_faces:
        faces = [(0, 1, 3, 2), (1, 5, 7, 3), (4, 6, 7, 5), (0, 2, 6, 4), (0, 4, 5, 1), (2, 3, 7, 6)]
        return corner_pnts, faces
    else:
        return corner_pnts

def project_points_to_2d(points, cam_K, cam_T):
    '''
    transform box corners to cam system
    :param points: N x 3 coordinates in world system
    :param cam_K: cam K matrix
    :param cam_T: 4x4 extrinsic matrix with open-gl setting. (http://www.songho.ca/opengl/gl_camera.html)
                  [[v1, v2, v3, T]
                   [0,  0,  0,  1,]]
                  where v1, v2, v3 corresponds to right, up, backward of a camera
    '''
    # transform to camera system
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = np.linalg.inv(cam_T).dot(points_h.T)
    points_cam = points_cam[:3]

    # transform to opencv system
    points_cam[1] *= -1
    points_cam[2] *= -1

    # delete those points whose depth value is non-positive.
    invalid_ids = np.where(points_cam[2] <= 0)[0]
    points_cam[2, invalid_ids] = 0.0001

    # project to image plane
    points_cam_h = points_cam / points_cam[2][np.newaxis]
    pixels = (cam_K.dot(points_cam_h)).T

    return pixels


def get_iou_cuboid(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """

    # 2D projection on the horizontal plane (z-x plane)
    polygon2D_1 = Polygon(
        [(cu1[0][2], cu1[0][0]), (cu1[1][2], cu1[1][0]), (cu1[5][2], cu1[5][0]), (cu1[4][2], cu1[4][0])])

    polygon2D_2 = Polygon(
        [(cu2[0][2], cu2[0][0]), (cu2[1][2], cu2[1][0]), (cu2[5][2], cu2[5][0]), (cu2[4][2], cu2[4][0])])

    # 2D intersection area of the two projections.
    intersect_2D = polygon2D_1.intersection(polygon2D_2).area

    # the volume of the intersection part of cu1 and cu2
    inter_vol = intersect_2D * max(0.0, min(cu1[2][1], cu2[2][1]) - max(cu1[0][1], cu2[0][1]))

    # the volume of cu1 and cu2
    vol1 = polygon2D_1.area * (cu1[2][1]-cu1[0][1])
    vol2 = polygon2D_2.area * (cu2[2][1]-cu2[0][1])

    # return 3D IoU
    return inter_vol / (vol1 + vol2 - inter_vol)


def write_data_to_hdf5(file_handle, name, data):
    if isinstance(data, (list, tuple)):
        if not len(data):
            file_handle.create_dataset(name,  data=h5py.Empty("i"))
        elif isinstance(data[0], (int, np.integer)):
            file_handle.create_dataset(name, shape=(len(data),), dtype=np.int32, data=np.array(data))
        elif isinstance(data[0], (float, np.float)):
            file_handle.create_dataset(name, shape=(len(data),), dtype=np.float32, data=np.array(data))
        elif isinstance(data[0], str):
            asciiList = [item.encode("ascii", "ignore") for item in data]
            file_handle.create_dataset(name, shape=(len(asciiList),), dtype='S10', data=asciiList)
        elif isinstance(data[0], (dict, list)):
            group_data = file_handle.create_group(name)
            for node_idx, node in enumerate(data):
                write_data_to_hdf5(group_data, str(node_idx), node)
        else:
            raise NotImplementedError
    elif isinstance(data, (int, np.integer)):
        file_handle.create_dataset(name, shape=(1,), dtype='i', data=data)
    elif isinstance(data, (float, np.float)):
        file_handle.create_dataset(name, shape=(1,), dtype='f', data=data)
    elif isinstance(data, str):
        dt = h5py.special_dtype(vlen=str)
        file_handle.create_dataset(name, shape=(1,), dtype=dt, data=data)
    elif isinstance(data, np.ndarray):
        file_handle.create_dataset(name, shape=data.shape, dtype=data.dtype, data=data)
    elif isinstance(data, type(None)):
        dt = h5py.special_dtype(vlen=str)
        file_handle.create_dataset(name, shape=(1,), dtype=dt, data='null')
    elif isinstance(data, dict):
        group_data = file_handle.create_group(name)
        for key, value in data.items():
            write_data_to_hdf5(group_data, key, value)
    else:
        raise TypeError('Unrecognized data type.')
    return