python main.py \
mode=generation \
start_deform=True \
data.dataset=3D-Front \
finetune=True \
weight=/mnt/ikarus/SceneSynthesis/outputs/3D-Front/train/2022-11-11/22-17-38/model_best.pth \
generation.room_type=bed \
data.split_dir=splits \
data.split_type=bed \
generation.phase=generation