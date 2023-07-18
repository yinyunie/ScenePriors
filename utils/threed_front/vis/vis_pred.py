#  Copyright (c) 9.2022. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
from utils.threed_front import Threed_Front_Config
import trimesh
import vtk
from utils.vis_base import VIS_BASE
import seaborn as sns
from utils.threed_front.tools.threed_future_dataset import ThreedFutureDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3D-FRONT gt sample.")
    parser.add_argument("--pred_file", type=str, default='outputs/3D-Front/generation/2022-10-10/20-32-01/vis/bed/sample_0_0.npz',
                        help="The directory of dumped results.")
    parser.add_argument("--use_retrieval", action='store_true',
                        help="If use 3D future models.")
    parser.add_argument("--former_dir", type=str,
                        help="The directory of dumped results. e.g., 2022-10-09-19-30-20")
    parser.add_argument('--path_to_pickled_3d_futute_models', type=str, help='pickled 3d-future dir from ATISS',
                        default='datasets/3D-Front/pickled_threed_future_model_%s.pkl')
    return parser.parse_args()


class VIS_3DFRONT_RESULT(VIS_BASE):
    def __init__(self, mesh_files, category_ids, class_names, **kwargs):
        super(VIS_3DFRONT_RESULT, self).__init__()
        self._cam_K = np.array([[256, 0, 256], [0, 256, 256], [0, 0, 1]])
        self.mesh_files = mesh_files
        self.class_ids = category_ids
        self.class_names = [class_names[cls_id] for cls_id in category_ids]
        self.cls_palette = np.array(sns.color_palette('hls', len(class_names)))

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        cam_loc = np.array([0, 6, 0])
        cam_fp = np.array([0, 0, 0])
        cam_up = np.array([1, 0, 0])
        fov_y = (2 * np.arctan((self.cam_K[1][2] * 2 + 1) / 2. / self.cam_K[1][1])) / np.pi * 180
        camera = self.set_camera(cam_loc, cam_fp, cam_up, fov_y=fov_y)
        camera.SetParallelProjection(True)
        camera.SetParallelScale(5)
        renderer.SetActiveCamera(camera)

        '''draw instance meshes'''
        for cls_id, cls_name, mesh_file in zip(self.class_ids, self.class_names, self.mesh_files):
            # draw meshes
            obj_actor = self.get_obj_actor(mesh_file)
            obj_actor.GetProperty().SetColor(self.cls_palette[cls_id])
            renderer.AddActor(obj_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, 10, -10), (-10, 10, -10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(0.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

def read_pred_data(pred_file, dataset_config, use_retrieval=False, objects_dataset=None):
    pred_data = np.load(pred_file)
    box3ds = np.concatenate([pred_data['centers'], pred_data['sizes']], axis=-1)
    box3ds = np.pad(box3ds, ((0, 0), (0, 1)))
    category_ids = pred_data['category_ids']
    mesh_vertices = pred_data['mesh_vertices']
    mesh_faces = pred_data['mesh_faces']

    save_mesh_dir = Path('./temp/generations/%s/%s/%s'%(room_type, 'retrieval' if args.use_retrieval else 'no_retrieval',current_time)).joinpath(pred_file.name[:-4])
    if not save_mesh_dir.exists():
        save_mesh_dir.mkdir(parents=True)

    inst_mesh_files = []
    for inst_id, (vertices, faces) in enumerate(zip(mesh_vertices, mesh_faces)):
        save_file = save_mesh_dir.joinpath('%d.obj' % inst_id)
        if not save_file.exists():
            color = color_palette[category_ids[inst_id]]
            if use_retrieval:
                vertices, faces = retrieval_model(vertices, category_ids[inst_id], objects_dataset, dataset_config)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, vertex_colors=color)
            mesh.export(save_file)
        inst_mesh_files.append(str(save_file))
    return {'box3ds': box3ds, 'category_ids': category_ids, 'mesh_files': inst_mesh_files}

def retrieval_model(source_vertices, cls_id, objects_dataset, dataset_config):

    query_label = dataset_config.label_names[cls_id]

    # ours new
    source_lbdb = source_vertices.min(axis=0)
    source_ubdb = source_vertices.max(axis=0)
    attach_to_floor = False
    if source_lbdb[1] < 0.3:
        attach_to_floor = True
        source_lbdb[1] = 0
    source_center = (source_lbdb + source_ubdb) / 2.

    query_vertices = source_vertices - source_center
    furniture, rot_mat = objects_dataset.get_closest_furniture_to_box(query_label, query_vertices, dataset_config.generic_mapping)

    furniture_mesh = trimesh.load(furniture.raw_model_path)
    key_vertices = furniture_mesh.vertices

    key_vertices = key_vertices * furniture.scale
    key_lbdb = key_vertices.min(axis=0)
    key_ubdb = key_vertices.max(axis=0)
    key_centroid = (key_lbdb + key_ubdb) / 2.
    key_vertices = key_vertices - key_centroid
    key_vertices = key_vertices.dot(rot_mat)

    key_vertices = key_vertices + source_center

    if attach_to_floor:
        lbdb = key_vertices.min(axis=0)
        key_vertices = key_vertices - [0, lbdb[1], 0]

    return key_vertices, furniture_mesh.faces


if __name__ == '__main__':
    args = parse_args()
    dataset_config = Threed_Front_Config()
    pred_file = Path(args.pred_file)
    room_type = pred_file.parent.name
    dataset_config.init_generic_categories_by_room_type(room_type)

    render_dir = pred_file.parents[1].joinpath('imgs')

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models % (room_type)
    )

    current_time = args.former_dir if args.former_dir is not None else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    color_palette = np.array(sns.color_palette('hls', len(dataset_config.label_names)))
    '''load gt and pred data'''
    if not pred_file.exists():
        raise FileNotFoundError('There is no such file.')
    '''read pred data'''
    pred_data = read_pred_data(pred_file, dataset_config, args.use_retrieval, objects_dataset)

    if pred_data is None:
        raise ValueError('pred_data is None.')

    '''visualize results'''
    # vis prediction
    viser = VIS_3DFRONT_RESULT(category_ids=pred_data['category_ids'], class_names=dataset_config.label_names,
                               mesh_files=pred_data['mesh_files'])
    viser.visualize()