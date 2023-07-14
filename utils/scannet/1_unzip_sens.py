#  Copyright (c) 7.2022. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
from utils.scannet import ScanNet_Config
from utils.scannet.tools.SensorData import SensorData
import zipfile


def write_frame_data(scene_path, frame_skip, sens_config):
    filename = scene_path.joinpath(scene_path.name + '.sens')

    depth_path = scene_path.joinpath('depth')
    color_path = scene_path.joinpath('color')
    pose_path = scene_path.joinpath('pose')
    intrinsic_path = scene_path.joinpath('intrinsic')

    if not pose_path.parent.exists():
        pose_path.parent.mkdir(parents=True)

    # load the data
    sys.stdout.write('loading %s...' % filename)
    sd = SensorData(filename)
    sys.stdout.write('loaded!\n')
    if sens_config['if_export_depth_images']['flag']:
        sd.export_depth_images(depth_path, frame_skip=frame_skip)
    if sens_config['if_export_color_images']['flag']:
        sd.export_color_images(color_path, frame_skip=frame_skip)
    if sens_config['if_export_poses']['flag']:
        sd.export_poses(pose_path, frame_skip=frame_skip)
    if sens_config['if_export_intrinsics']['flag']:
        sd.export_intrinsics(intrinsic_path)

def if_write(scene_path, sens_config):
    write_flag = False
    for if_label, item in sens_config.items():
        if not item['flag']:
            continue
        if not scene_path.joinpath(item['folder_name']).exists():
            write_flag = True
            break
    return write_flag

def export_data(scene_path):
    print('='*100)
    print('Processing: %s.' % str(scene_path))
    write_flag = if_write(scene_path, sens_config)
    if write_flag:
        write_frame_data(scene_path, frame_skip, sens_config)

    semantic_file = scene_path.joinpath(sem_inst_config['if_semantic']['folder_name'])
    if sem_inst_config['if_semantic']['flag'] and not semantic_file.exists():
        sem_zip_file = scene_path.joinpath(scene_path.name + '_2d-label-filt.zip')
        with zipfile.ZipFile(sem_zip_file, 'r') as zip_ref:
            zip_ref.extractall(scene_path)

    instance_file = scene_path.joinpath(sem_inst_config['if_instance']['folder_name'])
    if sem_inst_config['if_instance']['flag'] and not instance_file.exists():
        inst_zip_file = scene_path.joinpath(scene_path.name + '_2d-instance-filt.zip')
        with zipfile.ZipFile(inst_zip_file, 'r') as zip_ref:
            zip_ref.extractall(scene_path)
    print('Finished.')


if __name__ == '__main__':
    dataset_config = ScanNet_Config()
    sens_config = {'if_export_depth_images':
                       {'flag': False,
                        'folder_name': 'depth'},
                   'if_export_color_images':
                       {'flag': True,
                        'folder_name': 'color'},
                   'if_export_poses':
                       {'flag': True,
                        'folder_name': 'pose'},
                   'if_export_intrinsics':
                       {'flag': True,
                        'folder_name': 'intrinsic'}}
    sem_inst_config = {'if_semantic':
                           {'flag': True,
                            'folder_name': 'label-filt'},
                       'if_instance':
                           {'flag': True,
                            'folder_name': 'instance-filt'}}

    frame_skip = 15

    for scene_path in dataset_config.scene_paths:
        export_data(scene_path)
