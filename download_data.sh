# We defaulty put all data in the Train/datasets. You can put them anywhere but create softlinks under Train/datasets.

# We provide two way to download data. 1) Cloudstor; 2) Google Drive
# 1. Download from CloudStor:

# download part-fore
cd Train/datasets
mkdir DiverseDepth
cd DiverseDepth
wget https://cloudstor.aarnet.edu.au/plus/s/HNfpS4tAz3NePtU/download -O DiverseDepth_d.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/n5bOhKk52fXILp9/download -O DiverseDepth_rgb.zip
tar -xvf DiverseDepth_d.tar.gz
unzip DiverseDepth_rgb.zip


# download part_out, collected from DIML
cd ..
mkdir DIML_GANet
cd DIML_GANet
wget https://cloudstor.aarnet.edu.au/plus/s/xfNCkAAwGPbH1jH/download -O annotations.zip
wget https://cloudstor.aarnet.edu.au/plus/s/FsUa7zKTBHplj34/download -O DIML_depth.tat.gz
wget https://cloudstor.aarnet.edu.au/plus/s/yVerFac2ZbmIZqv/download -O DIML_rgb.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/66a6A2OLGzfOtC8/download -O DIML_sky.tar.gz
tar -xvf ./*.tat.gz
unzip annotations.zip


# download part_in, collected from taskonomy
cd ..
mkdir taskonomy
cd taskonomy
wget https://cloudstor.aarnet.edu.au/plus/s/Q4jqXt2YfqcGZvK/download -O annotations.zip
wget https://cloudstor.aarnet.edu.au/plus/s/EBv6jRp326zMlf6/download -O taskonomy_rgbs.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/t334giSOJtC97Uq/download -O taskonomy_ins_planes.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/kvLcrVSWfOsERsI/download -O taskonomy_depths.tar.gz
tar -xvf ./*.tar.gz
unzip annotations.zip


# The overview of data under Train/datasets are:
# -Train
# |--datasets
#    |--DiverseDepth
#       |--annotations
#       |--depths
#       |--rgbs
#    |--taskonomy
#       |--annotations
#       |--depths
#       |--rgbs
#       |--ins_planes
#    |--DiverseDepth
#       |--annotations
#       |--depth
#       |--rgb
#       |--sky_mask


