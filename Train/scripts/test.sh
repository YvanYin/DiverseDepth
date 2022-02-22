
export PYTHONPATH=../../Train:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

TIME=`date +%Y-%m-%d_%H-%M-%S`

LOG="./$TIME.txt"

python ../tools/test_diversedepth_nyu.py \
--dataroot ./datasets \
--batchsize 1 \
--load_ckpt /home/yvan/DeepLearning/Depth/DiverseDepth-github/DiverseDepth/datasets/epoch18_step70000.pth \
$1 2>&1 | tee $LOG
