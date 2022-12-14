# The code of paper EditVAE: Unsupervised Part-Aware Controllable 3D Point Cloud Shape Generation (https://arxiv.org/abs/2110.06679)


## repo included:
TreeGAN: https://github.com/jtpils/TreeGAN

Superquadrics Revisited: https://github.com/paschalidoud/superquadric_parsing

## ShapeNet data (following TreeGAN):
On the google drive: https://drive.google.com/file/d/1IoWIjcuGn0yhMe-c4WPTBIbjkYVM1YR4/view?usp=sharing

add the data in the path: data/datasetTreeGAN/

## Training:
All the training, generation, and visualizations are in file code/evae.py by running: CUDA_VISIBLE_DEVICES=0 python code/oVAEs/evae.py

The model of EditVAE is mainly in code/oVAEs/sVAEs_module.py

## Environment:
PyTorchEMD: https://github.com/daerduoCarey/PyTorchEMD

mayavi (for visualization): https://github.com/enthought/mayavi

torch

numpy

subprocess

mplot3d

sklearn

## Measurements:
re-implemented Minimum Matching Distance (MMD) and Coverage (COV)

Jensen-Shannon Divergence (JSD) is mainly from the code of rGAN: https://github.com/optas/latent_3d_points

## Citation:
If you found this work influential or helpful for your research, please consider citing:
```
@inproceedings{li2022editvae,
  title={EditVAE: Unsupervised Parts-Aware Controllable 3D Point Cloud Shape Generation},
  author={Li, Shidi and Liu, Miaomiao and Walder, Christian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={2},
  pages={1386--1394},
  year={2022}
}
```
