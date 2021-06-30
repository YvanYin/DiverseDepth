#### DiverseDepth Project
This project aims to improve the generalization ability of the monocular depth estimation method on diverse scenes. We propose a learning method and a diverse dataset, termed DiverseDepth, to solve this problem. 

This repository contains the source code of our paper:
1. [Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data](https://arxiv.org/abs/2002.00569).
2. [Wei Yin, Yfan Liu, Chunhua Shen, Virtual Normal: Enforcing Geometric Constraints for Accurate and Robust Depth Prediction](https://arxiv.org/abs/2103.04216).

## Some Results

![Any images online](./examples/any_imgs.jpg)
![Point cloud](./examples/pcd.png)

## Some Dataset Examples
![Dataset](./examples/dataset_examples.png)


****
## Hightlights
- **Generalization:** Our method demonstrates strong generalization ability on several zero-shot datasets.


****
## Installation
- Please refer to [Installation](./Installation.md).

## Datasets
We collect multi-source data to construct our DiverseDepth dataset. It consists of three parts:
Part-in:  contains 93838 images
Part-out: contains 120293 images
Part-fore: contains 109703 images
You can download them with the following method.

```
sh download_data.sh
```


## Quick Start (Inference)

1. Download the model weights
   * [ResNeXt50 backbone](https://cloudstor.aarnet.edu.au/plus/s/ixWf3nTJFZ0YE4q)
2. Prepare data. 
   * Move the downloaded weights to  `<project_dir>/` 
   * Put the testing RGB images to `<project_dir>/Minist_Test/test_images/`. Predicted depths and reconstructed point cloud are saved under `<project_dir>/Minist_Test/test_images/outputs`

3. Test monocular depth prediction. Note that the predicted depths are affine-invariant. 
```bash
export PYTHONPATH="<PATH to DiverseDepth>"
# run the ResNet-50
python ./Minist_Test/tools/test_depth.py --load_ckpt model.pth
 
```

## Training
The training code will be released soon.


### Citation
```
@article{yin2021virtual,
  title={Virtual Normal: Enforcing Geometric Constraints for Accurate and Robust Depth Prediction},
  author={Yin, Wei and Liu, Yifan and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2021}
}
@article{yin2020diversedepth,
  title={DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data},
  author={Yin, Wei and Wang, Xinlong and Shen, Chunhua and Liu, Yifan and Tian, Zhi and Xu, Songcen and Sun, Changming and Renyin, Dou},
  journal={arXiv preprint arXiv:2002.00569},
  year={2020}
}
```
### Contact
Wei Yin: wei.yin@adelaide.edu.au
