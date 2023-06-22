## Introduction 

This is the PyTorch implementation of **VMLoc**, a simple and efficient neural architecture for Multimodal Camera Localization.

## Setup

VMLoc provides support for multiple versions of Python and Torch, such as:
```
python==3.8 
pytorch==1.10.1 
torchvision==0.11.2 
```
```
python==2.7
pytorch==0.4.1
torchvision==0.2.0
```

## Data
We support the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and [Oxford RobotCar](http://robotcar-dataset.robots.ox.ac.uk/) datasets right now. You can also write your own PyTorch dataloader for other datasets and put it in the `data` directory.

### Special instructions for RobotCar:

1. For each sequence, you need to download the `stereo_centre`, `vo` and `gps` tar files from the dataset website. The directory for each 'scene' (e.g. `loop`) has .txt files defining the train/test_split.
 
2. To make training faster, we pre-processed the images using `data/process_robotcar.py`. This script undistorts the images using the camera models provided by the dataset, and scales them such that the shortest side is 256 pixels.

3. Pixel and Pose statistics must be calculated before any training. Use the `data/dataset_mean.py`, which also saves the information at the proper location. Rre-computed values for RobotCar and 7Scenes could be find [Here(7Scenes)](https://github.com/BingCS/AtLoc/tree/master/data/7Scenes) and [Here(RobotCar)](https://github.com/BingCS/AtLoc/tree/master/data/RobotCar).

## Running the code

### Training
The executable script is `training.py`. For example:

- Directly Concatenation on `chess` from `7Scenes`: 
```
python -u -m training --dataset 7Scenes --scene stairs --mode concatenate \
    --gpus 0 --color_jitter 0. --train_mean 0
```

- VMLoc on `chess` from `7Scenes`: 
```
python -u -m training --dataset 7Scenes --scene stairs --mode vmloc \
    --gpus 0 --color_jitter 0. --train_mean 0 --train_mask 50
```

- VMLoc on `redkitchen` from `7Scenes`: 
```
python -u -m training --dataset 7Scenes --scene redkitchen --mode vmloc \
    --gpus 0 --color_jitter 0. --train_mean 0 --train_mask 150
```

- VMLoc on `loop` from `RobotCar`: 
```
python -u -m training --dataset RobotCar --scene loop --mode vmloc \
    --gpus 0 --gamma -3.0 --color_jitter 0.7 --train_mean 0 --train_mask 500
```

The meanings of various command-line parameters are documented in train.py. The values of various hyperparameters are defined in `tools/options.py`.

### Inference
The trained models for partial experiments presented in the paper could be found in `record`. The inference script is `testing.py`. Here are some examples, assuming the models are downloaded in `logs`.

- VMLoc on `redkitchen` from `7Scenes`: 
```
python testing.py --mode vmloc --dataset 7Scenes --scene redkitchen \
    --seed 2019 --weights record/red_kitchen.pth.tar
```

## Citation
If you find this code useful for your research, please cite our paper

```
@inproceedings{zhou2021vmloc,
  title={Vmloc: Variational fusion for learning-based multimodal camera localization},
  author={Zhou, Kaichen and Chen, Changhao and Wang, Bing and Saputra, Muhamad Risqi U and Trigoni, Niki and Markham, Andrew},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={7},
  pages={6165--6173},
  year={2021}
}
```
## Acknowledgements
Our code partially builds on [MapNet](https://github.com/NVlabs/geomapnet), [PoseLstm](https://github.com/hazirbas/poselstm-pytorch), and [AtLoc](https://github.com/BingCS/AtLoc/tree/master).
