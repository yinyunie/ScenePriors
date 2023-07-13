#  Copyright (c) 9.2022. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import argparse
import h5py
import numpy as np
from collections import defaultdict
from utils.threed_front import Threed_Front_Config
from utils.tools import write_json, read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Process for training")
    parser.add_argument("--room_type",
                        default="living",
                        choices=["bed", "living", "dining", "library"],
                        help="The type of dataset filtering to be used.")
    return parser.parse_args()


def load_sample_hdf5(sample_file):
    with h5py.File(sample_file, "r") as sample_data:
        inst_h5py = sample_data['inst_info']
        room_uid = sample_data['room_uid'][0].decode('ascii')
        category_ids = []
        box2ds = []
        inst_marks = []
        for inst_id in inst_h5py:
            box2ds.append(inst_h5py[inst_id]['bbox2d'][:])
            category_ids.append(inst_h5py[inst_id]['category_id'][0])
            inst_marks.append(inst_h5py[inst_id]['inst_mark'][0].decode('ascii'))

    insts = {
        'room_uid': room_uid,
        'box2ds': box2ds,
        'category_ids': category_ids,
        'inst_marks': inst_marks}

    return insts

if __name__ == '__main__':
    args = parse_args()
    # initialize category labels and mapping dict for specific room type.
    dataset_config = Threed_Front_Config()
    dataset_config.init_generic_categories_by_room_type(args.room_type)

    output_dir = dataset_config.dump_dir_to_samples.joinpath(args.room_type)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # statistics of the area ratio for each
    area_ratio_per_cls = defaultdict(list)
    unique_inst_marks = defaultdict(set)

    for output_file in output_dir.iterdir():
        inst_info = load_sample_hdf5(output_file)

        # per sample pix occupancy ratio
        per_sample_stat = defaultdict(int)
        for category_id, box2d in zip(inst_info['category_ids'], inst_info['box2ds']):
            per_sample_stat[category_id] += (np.prod(box2d[2:]) / np.prod(dataset_config.image_size))

        for cls, ratio in per_sample_stat.items():
            area_ratio_per_cls[cls].append(ratio)

        # unique inst marks
        unique_inst_marks[inst_info['room_uid']] = unique_inst_marks[inst_info['room_uid']].union(inst_info['inst_marks'])

    '''conclude unique instance marks for each room'''
    out_unique_inst_marks = {}
    for key, item in unique_inst_marks.items():
        out_unique_inst_marks[key] = sorted(list(item))

    if dataset_config.unique_inst_mark_path.exists():
        exist_unique_inst_marks = read_json(dataset_config.unique_inst_mark_path)
        exist_unique_inst_marks.update(out_unique_inst_marks)
    else:
        exist_unique_inst_marks = out_unique_inst_marks

    write_json(dataset_config.unique_inst_mark_path, exist_unique_inst_marks)
