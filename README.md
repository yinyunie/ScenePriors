# Learning 3D Scene Priors with 2D Supervision [[Project]](https://yinyunie.github.io/sceneprior-page/)[[Paper]](https://arxiv.org/pdf/2211.14157.pdf)

[Yinyu Nie](https://yinyunie.github.io/), [Angela Dai](https://www.3dunderstanding.org/), [Xiaoguang Han](https://gaplab.cuhk.edu.cn/), [Matthias Nießner](https://niessnerlab.org/index.html)

in [CVPR 2023](https://cvpr2023.thecvf.com/)

---

**3D Scene Generation**

| <img src="resources/scene_gen/rendering_1.jpg" width="500"> | <img src="resources/scene_gen/rendering_2.jpg" width="500"> |<img src="resources/scene_gen/rendering_3.jpg" width="500"> | <img src="resources/scene_gen/rendering_4.jpg" width="500"> |
|:-----------------------------------------------------------:|:-----------------------------------------------------------:|:-----------------------------------------------------------:|:-----------------------------------------------------------:| 


**Single View Reconstruction**

|                       Input                       |                        Pred                         |                       Input                       |                       Pred                       |
|:-------------------------------------------------:|:---------------------------------------------------:|:-------------------------------------------------:|:------------------------------------------------:|
| <img src="resources/svr/1/input.jpg" width="500"> |  <img src="resources/svr/1/ours.jpg" width="500">   | <img src="resources/svr/2/input.jpg" width="500"> | <img src="resources/svr/2/ours.jpg" width="500"> |
 ---

## Install
Our codebase is developed under Ubuntu 20.04 with PyTorch 1.12.1.
1. We recommend to use [conda]() to deploy our environment by
    ```commandline
    cd ScenePriors
    conda create env -f environment.yml
    conda activate sceneprior
    ```

2. Install [Fast Transformers](https://fast-transformers.github.io/) by
   ```commandline
   cd external/fast_transformers
   python setup.py build_ext --inplace
   cd ../..
   ```

3. Please follow [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install the prerequisite libraries for [PyTorch3D](https://pytorch3d.org/). Then install PyTorch3D from our local clone by
   ```commandline
   cd external/pytorch3d
   pip install -e .
   cd ../..
   ```
   *Note: After installed all prerequisite libraries in [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), please do not install prebuilt binaries for PyTorch3D.*  
---

## Data Processing
### 3D-Front data processing
1. Apply \& Download the [3D-Front](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset and link them to the local directory as follows:
   ```
   datasets/3D-Front/3D-FRONT
   datasets/3D-Front/3D-FRONT-texture
   datasets/3D-Front/3D-FUTURE-model
   ```
   
2. Render 3D-Front scenes following my [rendering pipeline](https://github.com/yinyunie/BlenderProc-3DFront) and link the rendering results (in `renderings` folder) to
   ```
   datasets/3D-Front/3D-FRONT_renderings_improved_mat
   ```
   *Note: you can comment out `bproc.renderer.enable_depth_output(activate_antialiasing=False)` in `render_dataset_improved_mat.py` since we do not need depth information.*

3. Preprocess 3D-Front data by
   ```commandline
   python utils/threed_front/1_process_viewdata.py --room_type ROOM_TYPE --n_processes NUM_THREADS
   python utils/threed_front/2_get_stats.py --room_type ROOM_TYPE
   ```
   * The processed data for training are saved in `datasets/3D-Front/3D-FRONT_samples`.
   * We also parsed and extracted the 3D-Front data for visualization into `datasets/3D-Front/3D-FRONT_scenes`.
   * `ROOM_TYPE` can be `'bed'`(bedroom) or `'living'`(living room).
   * You can set `NUM_THREADS` to your CPU core number for parallel processing.

4. Visualize processed data for verification by (optional)
   ```commandline
   python utils/threed_front/vis/vis_gt_sample.py --scene_json SCENE_JSON_ID --room_id ROOM_ID --n_samples N_VIEWS 
   ```
   * `SCENE_JSON_ID` is the ID of a scene, e,g, `6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9`.
   * `ROOM_ID` is the room ID in this scene, e.g., `MasterBedroom-2679`.
   * `N_VIEWS` is the number views to visualize., e.g. `12`.
   
   If everything goes smooth, there will pop five visualization windows as follows.

|           <div style="width:200px">RGB</div>            |                        <div style="width:200px">Semantics</div>                        |                        <div style="width:200px">Instances</div>                         |                            <div style="width:200px">3D Boxes</div>                             |           <div style="width:200px">CAD Models</div>            |
|:-------------------------------------------------------:|:--------------------------------------------------------------------------------:|:--------------------------------------------------------:|:---------------------------------------------------------------:|:--------------------------------------------------------------:|
| <img src="resources/visualization/rgb.jpg" width="300"> |             <img src="resources/visualization/sem.jpg" width="300">              | <img src="resources/visualization/inst.jpg" width="300"> | <img src="resources/visualization/3dboxesproj.jpg" width="300"> | <img src="resources/visualization/CAD_models.png" width="300"> |

*Note: X server is required for 3D visualization.*

