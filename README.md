#### DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data.

This repository contains the source code of our paper:
[Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data](https://arxiv.org/abs/2002.00569).

## Some Results

![Any images online](./examples/any_imgs.jpg)
![Point cloud](./examples/pcd.png)

## Some Dataset Examples
![Dataset](./examples/dataset_examples.png)


****
## Hightlights
- **Generalization:** We have tested on several zero-shot datasets to test the generalization of our method. 



****
## Installation
- Please refer to [Installation](./Installation.md).

## Datasets
We collect multiply source data to construct our DiverseDepth dataset, including crawling online stereoscopic images, images from DIML and Taskonomy. These three parts form the foreground parts (Part-fore), outdoor scenes (Part-out) and indoor scenes (Part-in) of our dataset. 
The size of three parts are:
Part-in:  contains 93838 images
Part-out: contains 120293 images
Part-fore: contains 109703 images
 We will release the dataset as soon as possible. 
  
## Model Zoo
- ResNext50_32x4d backbone, trained on DiverseDepth dataset, download [here](https://cloudstor.aarnet.edu.au/plus/s/ixWf3nTJFZ0YE4q)


  
## Inference

```bash
# Run the inferece on NYUDV2 dataset
 python  ./tools/test_diversedepth_nyu.py \
		--dataroot    ./datasets/NYUDV2 \
		--dataset     nyudv2 \
		--cfg_file     lib/configs/resnext50_32x4d_diversedepth_regression_vircam \
		--load_ckpt   ./model.pth 
		
# Test depth predictions on any images, please replace the data dir in test_any_images.py
 python  ./tools/test_any_diversedepth.py \
		--dataroot    ./ \
		--dataset     any \
		--cfg_file     lib/configs/resnext50_32x4d_diversedepth_regression_vircam \
		--load_ckpt   ./model.pth 
```
If you want to test the kitti dataset, please see [here](./datasets/KITTI/README.md)



### Citation
```
@inproceedings{Yin2019enforcing,
  title={DiverseDepth: Affine-invariant Depth Prediction Using Diverse Data},
  author={Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun},
  booktitle= {arxiv: 2002.00569},
  year={2020}
}
```
### Contact
Wei Yin: wei.yin@adelaide.edu.au
