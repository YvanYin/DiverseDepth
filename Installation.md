## Installation

###Requirements
- PyTorch >= 1.1.0
- torchvision >= 0.2.1
- matplotlib
- opencv-python
- dill
- scipy
- yaml

### Step-by-step installation
```bash
# Firstly, your conda is setup properly with the right environment for that

conda create --n DiverseDepth python=3.7
conda activate DiverseDepth


# basic packages
conda install matplotlib dill pyyaml opencv scipy 

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install -c pytorch==1.4.0 torchvision=0.5.0 cudatoolkit=10.0
```
