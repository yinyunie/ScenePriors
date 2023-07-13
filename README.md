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
3. 


